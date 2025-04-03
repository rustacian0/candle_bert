from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from transformers import BertJapaneseTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
# BertJapaneseTokenizerの代わりにAutoTokenizerを使用
tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
from transformers import BertJapaneseTokenizer
original_tokenizer = BertJapaneseTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
tokenizer = Tokenizer(WordPiece(vocab=original_tokenizer.vocab, unk_token=original_tokenizer.unk_token))

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
tokenizer.save("tokenizer.json")