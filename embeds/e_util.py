#Don't filter anything (because we have their embeddings based on link-pred methods)
def loadAllrelsTotal(path):
    ret = []
    with open(path) as fin:
        first = True
        for line in fin:
            if not first:
                ret.append(line.split("\t")[0])
            first = False
    return ret

# Get the longest phrase from the CCG relation.
# relstr: string of format '(smile.1,smile.like.2)'
# returns: 'smile like'
def getPhraseFromCCGRel(relstr):
    # print relstr
    negated = False

    # Check for negation in front: "NEG__(smile.1,smile.like.2)"
    if relstr.startswith("NEG__"):
        relstr = relstr[5:]
        negated = True
    # Check for other prefixes
    elif relstr.find("__(") != -1:
        st = relstr.find("__(")
        relstr = relstr[(st + 3):]

    # Remove parentheses. Split on comma.
    pair = relstr[1:-1].split(",")
    if len(pair) != 2:
        # print "ERROR Bad relation string: ", relstr
        return ""

    # Removing trailing .1 or .2
    pred1 = pair[0][0:pair[0].rfind('.')]
    pred2 = pair[1][0:pair[1].rfind('.')]

    # Replace periods with spaces. Return the longest!
    phrase1 = pred1.replace('.', ' ')
    phrase2 = pred2.replace('.', ' ')
    if len(phrase2) >= len(phrase1):  # keep the longest
        phrase1 = phrase2
    if negated:  # check for negation
        phrase1 = "not " + phrase1
    return phrase1

# "be_the_need_of"
def getPhraseFromOpenIERel(relstr):
    phrase = relstr.replace('_', ' ')
    return phrase