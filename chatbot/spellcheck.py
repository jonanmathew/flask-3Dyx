import enchant
import re


def spell(sentence):
    engdict = enchant.Dict("en_US")
    misspelled = []
    suggestions = []
    # split sentence into individual words and punctuation marks
    words = re.findall(r"[\w']+|[.,!?;]", sentence)
    for word in words:
        if not engdict.check(word):
            misspelled.append(word)
            suggestions.append(engdict.suggest(word)[:5])
    if not misspelled:
        return "No misspelled words found."
    else:
        output = "The misspelled words are: \n"
        for i, word in enumerate(misspelled):
            output += word
            output += " (Suggestions: "
            if suggestions[i]:
                output += ", ".join(suggestions[i])
            else:
                output += "None"
            output += ")"
            if i != len(misspelled) - 1:
                output += "\n "
        return (output)
