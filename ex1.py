from datasets import load_dataset
from transformers import AutoConfig , AutoTokenizer , AutoModel




def main():

    raw_dataset = load_dataset('conll 2012 _ontonotesv 5 ', 'english_v 12')


if __name__ == "__main__":
    main()