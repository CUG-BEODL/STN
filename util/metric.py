from dataset import Label

def compute_sparse_accuracy(pre_dict, test_dict):
    PsLs = 0
    PsLc = 0
    PcLs = 0
    PcLc = 0

    for p_key in pre_dict:
        for r_key in test_dict:
            if p_key == r_key:
                pre_empty = len(pre_dict[p_key]) == 0
                test_empty = len(test_dict[r_key]) == 0
                if pre_empty and test_empty:
                    PsLs += 1
                elif not pre_empty and not test_empty:
                    PcLc += 1
                elif pre_empty and not test_empty:
                    PsLc += 1
                else:
                    PcLs += 1

    spa_acc = (PsLs + PcLc) / (PsLs + PsLc + PcLs + PcLc) if (PsLs + PsLc + PcLs + PcLc) > 0 else 0
    spa_ppv = (PsLs) / (PsLs + PcLs) if (PsLs + PcLs) > 0 else 0
    spa_tpr = (PsLs) / (PsLs + PcLc) if (PsLs + PcLc) > 0 else 0
    spa_tnr = (PcLc) / (PcLs + PcLc) if (PcLs + PcLc) > 0 else 0
    spa_f1 = (2*spa_ppv*spa_tpr) / (spa_ppv+spa_tpr) if (spa_ppv+spa_tpr) > 0 else 0

    return PsLs, PsLc, PcLs, PcLc, round(spa_acc, 3), round(spa_ppv, 3), round(spa_tpr, 3), round(spa_tnr, 3), round(spa_f1, 3)

def compute_temporal_accuracy(pre_dict, test_dict, class_dict, train1, train2, block_size):
    correct = 0
    for p_key in pre_dict:
        for r_key in test_dict:
            if p_key == r_key and (len(pre_dict[p_key]) or len(test_dict[r_key])):
                ls_p = pre_dict[p_key]
                ls_r = []  
                for lst in test_dict[r_key]:
                    ls_r.append(lst[0])
                    ls_r.append(lst[1])

                for i in range(block_size):
                    if i not in ls_p and i not in ls_r:
                        correct += 1 
                    elif i in ls_p and i in ls_r:
                        if r_key in train1 and class_dict[p_key] == 1:
                            correct += 1
                        elif r_key in train2 and class_dict[p_key] == 2:
                            correct += 1

    total = block_size * len(test_dict)
    acc_time = correct / total if total > 0 else 0
    return round(acc_time, 3)




