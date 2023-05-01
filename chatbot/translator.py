import googletrans
from googletrans import Translator
import json

def translator(text,dest):
    value=""
    c=0
    langdict=googletrans.LANGUAGES
    dest=dest.lower()
    for i in langdict:
        if langdict[i]==dest:
            value=[i][0]
            c=1
            break
    if c==0:
        return("\nLanguage not found\n")
    else:    
        translator = Translator()
        srcdict=translator.detect(text)
        srcdict=srcdict.__dict__
        src=srcdict["lang"]
        trans = translator.translate(text,src=src,dest=value) 
        ans=trans.text
        return(ans)