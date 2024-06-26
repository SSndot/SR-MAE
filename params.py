import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--seed', default=19260817, type=int, help='random seed')
    parser.add_argument('--data', default='books', type=str, help='name of dataset')
    parser.add_argument('--epoch', default=20, type=int, help='number of training epochs')
    parser.add_argument('--trn_batch', default=256, type=int, help='batch size training')
    parser.add_argument('--tst_batch', default=256, type=int, help='batch size for testing')
    parser.add_argument('--con_batch', default=2048, type=int, help='batch size for reconstruction task')
    parser.add_argument('--test_frequency', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--max_seq_len', default=50, type=int, help='maximnm number of items in an user sequence')
    parser.add_argument('--num_reco_neg', default=40, type=int, help='number of negative items for reconstruction task')
    parser.add_argument('--reg', default=1e-6, type=float, help='weight decay regularizer')
    parser.add_argument('--ssl_reg', default=1e-2, type=float, help='contrastive regularizer')
    parser.add_argument('--latdim', default=32, type=int, help='embedding size')
    parser.add_argument('--mask_depth', default=3, type=int, help='k steps for generating transitional path')
    parser.add_argument('--path_prob', default=0.5, type=float, help='random walk sample probability')
    parser.add_argument('--num_attention_heads', default=4, type=int, help='number of heads in attention')
    parser.add_argument('--num_gcn_layers', default=2, type=int, help='number of gcn layers')
    parser.add_argument('--num_trm_layers', default=2, type=int, help='number of gcn layers')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--num_mask_cand', default=50, type=int, help='number of seeds in patch masking')
    parser.add_argument('--mask_steps', default=10, type=int, help='steps to train on the same sampled graph')
    parser.add_argument('--eps', default=0.2, type=float, help='scaled weight for task-adaptive function')
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.3, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p")
    parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
    return parser.parse_args()


args = parse_args()
