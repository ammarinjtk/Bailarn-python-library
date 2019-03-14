def custom_matric(expect_word,retrieved):
    retrieved = set([e[0] for e in retrieved])
    relevant = set(expect_word)
    ans={}
    ans['precision'] = len(retrieved.intersection(relevant))/len(retrieved)
    ans['recall'] =len(retrieved.intersection(relevant))/len(relevant)
    if ans['precision'] == 0 or ans['recall'] == 0:
        ans['f1'] = 0
        return ans
    ans['f1'] = 2 * (ans['precision']*ans['recall'])/(ans['precision']+ans['recall'])
    return ans