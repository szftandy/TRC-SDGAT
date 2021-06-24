import numpy as np

def count_results(log_dir):
    fdr, tpr, fpr, shd, nnz, cor_e = [],[],[],[],[],[]
    with open(log_dir, 'r') as f:
        contents = f.readlines()
        for line in contents:
            if 'after  pruning' in line:
                info = line.rstrip().split(': ')[-1].split(', ')
                fdr_val = float(info[0][4:])
                fdr.append(fdr_val)
                tpr.append(float(info[1][4:]))
                fpr.append(float(info[2][4:]))
                shd.append(int(info[3][4:]))
                nnz_val = int(info[4][4:])
                nnz.append(nnz_val)
                cor_e.append(round((1-fdr_val)*nnz_val))
    return fdr, tpr, fpr, shd, nnz, cor_e

def best_result(log_dir, exp):
    fdr, tpr, fpr, shd, nnz, cor_e = count_results(log_dir)
    if exp >= 4:
        entry = np.stack([shd, cor_e, nnz])
        min_shd_ind = np.where(entry[0] == np.amin(entry[0]))[0]
        filter_entry = entry[:, min_shd_ind]
        max_cor_ind = np.argmax(filter_entry[1])
        shd, cor_e, nnz = filter_entry[:, max_cor_ind]
        return shd, cor_e, nnz
        
res = best_result('training.log', 5)