from tqdm import tqdm

from src.datasets.dataset_splitter import DatasetSplitter
from src.datasets.ppi_graph_builder import PPIGraphBuilder
from src.datasets.ppi_network_parser import PPINetworkParser
from src.datasets.pretrained_protein_feature_encoder import PretrainedProteinFeatureEncoder
from src.datasets.protein_sequence_store import ProteinSequenceStore
from src.datasets.sequence_vector_encoder import SequenceVectorEncoder


class GNN_DATA:
    def __init__(self, ppi_path, exclude_protein_path=None, max_len=2000, skip_head=True, p1_index=0, p2_index=1,
                 label_index=2, graph_undirection=True, bigger_ppi_path=None):
        self.ppi_path = ppi_path
        self.bigger_ppi_path = bigger_ppi_path
        self.max_len = max_len
        self.protein_dict = {}
        self.protein_dict_origin = {}
        self.pro_go_edge_type = []
        self.pro_go_edge = []

        parser = PPINetworkParser(skip_head=skip_head, p1_index=p1_index, p2_index=p2_index, label_index=label_index)
        parsed_network = parser.parse(
            ppi_path=ppi_path,
            exclude_protein_path=exclude_protein_path,
            bigger_ppi_path=bigger_ppi_path,
            graph_undirection=graph_undirection,
        )
        self.ppi_list = parsed_network['ppi_list']
        self.origin_ppi_list = parsed_network['origin_ppi_list']
        self.ppi_dict = parsed_network['ppi_dict']
        self.ppi_label_list = parsed_network['ppi_label_list']
        self.protein_name = parsed_network['protein_name']
        self.node_num = parsed_network['node_num']
        self.edge_num = parsed_network['edge_num']

    def get_protein_aac(self, pseq_path):
        self.pseq_path = pseq_path
        self.pseq_dict, self.protein_len = ProteinSequenceStore().load(pseq_path)

    def embed_normal(self, seq, dim):
        return SequenceVectorEncoder(max_len=self.max_len)._pad_or_trim(seq, dim)

    def vectorize(self, vec_path):
        self.acid2vec, self.dim, self.pvec_dict = SequenceVectorEncoder(max_len=self.max_len).vectorize(
            self.pseq_dict,
            vec_path,
        )

    def _pretrained_key(self, protein_name):
        return PretrainedProteinFeatureEncoder.pretrained_key(protein_name)

    def _sequence_embedding(self, seq, dim=512):
        return PretrainedProteinFeatureEncoder.sequence_embedding(seq, dim=dim)

    def pretrained_emb_init(self, pre_emb_path):
        self.pretrained_emb_dict = PretrainedProteinFeatureEncoder.load_or_fallback(pre_emb_path, self.pseq_dict)

    def get_feature_pretrained(self, pseq_path, pre_emb_path):
        self.get_protein_aac(pseq_path)
        self.pretrained_emb_init(pre_emb_path)
        self.protein_dict = PretrainedProteinFeatureEncoder.build_protein_features(
            self.protein_name,
            self.pretrained_emb_dict,
        )
        print("pretrained protein feature num: {}".format(len(self.protein_dict)))

    def get_feature_origin(self, pseq_path, vec_path):
        self.get_protein_aac(pseq_path)
        self.vectorize(vec_path)
        self.protein_dict_origin = {
            protein_name: self.pvec_dict[protein_name]
            for protein_name in tqdm(self.protein_name.keys())
        }

    def get_connected_num(self):
        self.ufs = PPIGraphBuilder.connected_components_count(self.node_num, self.ppi_list)

    def generate_data(self):
        graph_data = PPIGraphBuilder.build(
            node_num=self.node_num,
            protein_names=self.protein_name,
            ppi_list=self.ppi_list,
            ppi_label_list=self.ppi_label_list,
            protein_features=self.protein_dict,
            support_features=self.protein_dict_origin,
        )
        self.data, self.ufs, self.edge_index, self.edge_attr, self.mul_type, self.x, self.x_origin = graph_data

    def split_dataset(self, train_valid_index_path, test_size=0.2, random_new=False, mode='random'):
        self.ppi_split_dict = DatasetSplitter.split(
            ppi_list=self.ppi_list,
            edge_num=self.edge_num,
            train_valid_index_path=train_valid_index_path,
            test_size=test_size,
            random_new=random_new,
            mode=mode,
        )
