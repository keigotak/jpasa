from transformers import BertJapaneseTokenizer

class BertJapaneseTokenizerFast(BertJapaneseTokenizer):
  def __call__(self,text,text_pair=None,return_offsets_mapping=True,**kwargs):
    v=super().__call__(text=text,text_pair=text_pair,return_offsets_mapping=False,**kwargs)
    if return_offsets_mapping:
      import tokenizations
      if type(text)==str:
        z=zip([v["input_ids"].squeeze(0)],[text],[text_pair] if text_pair else [""])
      else:
        z=zip(v["input_ids"].squeeze(0),text,text_pair if text_pair else [""]*len(text))
      w=[]
      for a,b,c in z:
        a2b,b2a=tokenizations.get_alignments(self.convert_ids_to_tokens(a),b+c)
        x=[]
        for i,t in enumerate(a2b):
          if t==[]:
            s=(0,0)
            if a[i]==self.unk_token_id:
              j=[[-1]]+[t for t in a2b[0:i] if t>[]]
              k=[t for t in a2b[i+1:] if t>[]]+[[len(b+c)]]
              s=(j[-1][-1]+1,k[0][0])
          elif t[-1]<len(b):
            s=(t[0],t[-1]+1)
          else:
            s=(t[0]-len(b),t[-1]-len(b)+1)
          x.append(s)
        w.append(list(x))
      v["offset_mapping"]=w[0] if type(text)==str else w
    return v

def test_case1():
  tokenizer = BertJapaneseTokenizerFast.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
  token_ids = tokenizer('多くの旗指物が揺れ、あるものは倒れて、足軽たちの陣笠の向こうに消えていた。', return_tensors='pt')
  print(token_ids)

if __name__=='__main__':
  test_case1()