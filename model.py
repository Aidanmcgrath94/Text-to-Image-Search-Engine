import torch
from torch import nn
import torch.nn.functional as F
from config import config

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.caption_hidden = nn.Linear(config['word_dimension'], config['hidden_dimension'])
        self.image_hidden = nn.Linear(config['image_dimension'], config['hidden_dimension'])
        self.predict1 = nn.Linear(config['hidden_dimension'], config['model_dimension']) 
        self.predict2 = nn.Linear(config['hidden_dimension'], config['model_dimension']) 
        self.dropout = nn.Dropout(p=config['dropout_keep'])
        self.bn = nn.BatchNorm1d(config['model_dimension'])

    def forward(self, anc, pos, neg):
        return self.forward_image(anc), self.forward_caption(pos), self.forward_caption(neg)

    def forward_caption(self, x):
        x = self.flatten(x)
        x = F.relu(self.caption_hidden(x))
        x = self.dropout(x)
        x = self.predict1(x)
        # batch norm
        # L2 norm
        # x = self.bn(x)
        x = torch.nn.functional.normalize(x)
        return x
    
    def forward_image(self, y):
        y = self.flatten(y)
        y = F.relu(self.image_hidden(y))
        y = self.dropout(y)
        y = self.predict2(y)
        # batch norm
        # L2 norm
        # y = self.bn(y)
        y = torch.nn.functional.normalize(y)
        return y

    def save(self, id):
        torch.save(self.state_dict(),dir_path+'models/'+str(id)+'_'+config['model_name']+'.pth')
        return

    def evaluate(self, test):
        self.eval()
        test_embed = {}      
        for k, v in test.items():
            test_image = torch.FloatTensor(v[0])
            test_cap = torch.FloatTensor([v[1]])
            ti = self.forward_image(test_image)
            tc = self.forward_caption(test_cap)
            test_embed[k] = [ti, tc]

        knn_rec = {}
        idx_eval = 0
        len_eval = len(test_embed)

        t1, t5, t10, t100 = 0, 0, 0, 0
        for k, v in test_embed.items():
          idx_eval+=1
          eval_data = {}
          sys.stdout.write("\r	* Evaluation progress = {0}%".format(round(((float(idx_eval)/len_eval)*100),2)))
          eval_cap = torch.FloatTensor(v[1])
          for sk, sv in test_embed.items():
            eval_image = torch.FloatTensor(sv[0])
            #dist = pdist(eval_cap, eval_image)
            dist = torch.cdist(eval_cap, eval_image, p=2)
            eval_data[sk] = dist

          kt1 = 1 if k in list(dict(sorted(eval_data.items(), key=itemgetter(1))[:1]).keys()) else 0
          kt5 = 1 if k in list(dict(sorted(eval_data.items(), key=itemgetter(1))[:5]).keys()) else 0
          kt10 = 1 if k in list(dict(sorted(eval_data.items(), key=itemgetter(1))[:10]).keys()) else 0
          kt100 = 1 if k in list(dict(sorted(eval_data.items(), key=itemgetter(1))[:100]).keys()) else 0

          knn_rec[k] = [kt1, kt5, kt10, kt100]
          #if 1 in kt1: break

        i_r1, i_r5, i_r10, i_r100 = 0, 0, 0, 0
        for k, x in knn_rec.items():
            i_r1 += x[0]
            i_r5 += x[1]
            i_r10 += x[2]
            i_r100 += x[3]

        r_1 = round(i_r1/len(eval_data)*100,2)#*100
        r_5 = round(i_r5/len(eval_data)*100,2)#*100
        r_10 = round(i_r10/len(eval_data)*100,2)#*100
        r_100 = round(i_r100/len(eval_data)*100,2)#*100

        print("\n	* Recall@K: R@1: %.1f, R@5: %.1f, R@10: %.1f, R@100: %.1f" % (p_1, p_5, p_10, p_100))
        self.save('model')
        self.train()
        return knn_rec

