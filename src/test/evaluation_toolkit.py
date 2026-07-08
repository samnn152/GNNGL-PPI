import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from lime import lime_tabular
from sklearn.metrics import f1_score

from src.test.model_evaluator import ModelEvaluator


class EvaluationToolkit:
    @staticmethod
    def extract_features(model, valid_edge_id, graph, target_layer):
        activations = {}

        def hook_fn(module, input, output):
            activations[target_layer] = output.detach()

        hook = model._modules.get(target_layer).register_forward_hook(hook_fn)
        model(graph.x, graph.edge_index, valid_edge_id, graph.edge_attr, graph)
        feature_map = activations[target_layer]
        hook.remove()
        return feature_map

    @staticmethod
    def normalize_correlation(tensor_subset1, tensor_subset2):
        correlation_matrix = torch.matmul(tensor_subset1, tensor_subset2.t())
        return 2 * (correlation_matrix - correlation_matrix.min()) / (
                correlation_matrix.max() - correlation_matrix.min()) - 1

    @classmethod
    def visualize_features(cls, feature_map, node_id, layer):
        print(feature_map.shape)
        selected_samples = feature_map[:10, :]
        indices1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        indices2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        tensor_subset1 = feature_map[indices1]
        tensor_subset2 = feature_map[indices2]
        cls.normalize_correlation(tensor_subset1, tensor_subset2)
        correlation_matrix = np.corrcoef(selected_samples.detach().numpy())

        sns.set(rc={'figure.figsize': (10, 8), 'figure.dpi': 200})
        heatmap = sns.heatmap(correlation_matrix, cmap='viridis', xticklabels=indices1,
                              yticklabels=indices2)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0, horizontalalignment='right')
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
        plt.show()
        plt.savefig('/home/newdisk1/mff/mul_kg/MASSA-master/Multimodal_downstream/GNN-PPI/ppi_protein_go/'
                    'Result/NE/N/' + layer + '.png')
        plt.close()

    @classmethod
    def permutation_importance(cls, model, graph, test_mask, device, metric_fn=f1_score, num_permutations=100):
        baseline_score = metric_fn(graph.edge_attr_1[test_mask].cpu().data,
                                   ModelEvaluator.test(model, graph, test_mask, device),
                                   average='micro')
        permuted_scores = []
        for feature_idx in range(20):
            permuted_data = graph.x.clone()
            permuted_data[:, feature_idx] = torch.rand_like(permuted_data[:, feature_idx])
            graph.x = permuted_data
            permuted_score = metric_fn(graph.edge_attr_1[test_mask].cpu().data,
                                       ModelEvaluator.test(model, graph, test_mask, device),
                                       average='micro')
            permuted_scores.append(permuted_score)

        importance = baseline_score - np.mean(permuted_scores)
        importances = baseline_score - np.array(permuted_scores)
        top_k_indices = np.argsort(importances)[-10:][::-1]
        top_k_importances = importances[top_k_indices]

        print(f"Top {10} features:")
        for idx in top_k_indices:
            print(f"Feature {idx}: Importance = {importances[idx]}")

        plt.bar(range(len(top_k_importances)), top_k_importances)
        plt.xticks(range(len(top_k_importances)), top_k_indices)
        plt.xlabel('Feature Index')
        plt.ylabel('Permutation Importance')
        plt.show()
        plt.savefig(
            '/home/newdisk1/mff/mul_kg/MASSA-master/Multimodal_downstream/GNN-PPI/ppi_protein_go/Result/PRE/permutation_importance.png')
        plt.show()
        return importance

    @staticmethod
    def explain_instance(model, graph, valid_edge_id, target_class, device):
        explainer = lime_tabular.LimeTabularExplainer(graph.x.cpu().numpy(), mode="classification")
        original_features = graph.x.cpu().numpy()
        data_row = original_features[0, :]
        output = model(graph.x, graph.edge_index, valid_edge_id, graph.edge_attr, graph)
        print(data_row.shape)
        explanation = explainer.explain_instance(data_row, output, num_features=2, top_labels=0)
        local_features, weights = explanation.as_list(target_class)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=weights, y=local_features, palette="viridis")
        plt.title(f"LIME Explanation for Class {target_class}")
        plt.xlabel("Weight")
        plt.ylabel("Feature")
        plt.show()
        plt.savefig(
            '/home/newdisk1/mff/mul_kg/MASSA-master/Multimodal_downstream/GNN-PPI/ppi_protein_go/Result/LIME/1.png')
        plt.close()
