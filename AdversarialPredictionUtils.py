import torch.nn as nn
from ap_perf import PerformanceMetric, MetricLayer

class LogisticRegressionClassifier(nn.Module):
    def __init__(self, n_features=2):
        super().__init__()
        self.linear_transform_layer = nn.Linear(
            in_features=n_features,
            out_features=1,
            bias=True)
        self.double()
        
    def forward(self, x):
        y_beforesigmoid_N_ = self.linear_transform_layer.forward(x)
        y_logproba_N_ = nn.functional.logsigmoid(y_beforesigmoid_N_)
        return y_logproba_N_.squeeze() 


# Recall given precision metric
class RecallGvPrecision(PerformanceMetric):
    def __init__(self, th):
        self.th = th

    def define(self, C):
        return C.tp / C.ap   

    def constraint(self, C):
        return (C.tp / C.pp) >=self.th



# if __name__=='__main__':
#     #create toy data
#     # generate 225 positive data points
#     n_P = [30, 60, 30]
#     # n_P = [60, 120, 60]
#     P = 3
#     prng = np.random.RandomState(101)

#     # the first 25 data points come from mean 2-D mvn with mean [1, 2.5] and next 200 come from
#     # 2-D mvn with mean [1, 1]
#     mu_PD = np.asarray([
#         [0.7, 2.5],
#         [0.7, 1.0],
#         [0.7, 0.0]])

#     cov_PDD = np.vstack([
#         np.diag([0.06, 0.1])[np.newaxis,:],
#         np.diag([0.1, 0.1])[np.newaxis,:],
#         np.diag([0.06, 0.06])[np.newaxis,:],
#         ])

#     xpos_list = list()
#     for p in range(P):
#         x_ND = prng.multivariate_normal(mu_PD[p], cov_PDD[p], size=n_P[p])
#         xpos_list.append(x_ND)
#     x_pos_ND = np.vstack(xpos_list)
#     y_pos_N  = np.ones(x_pos_ND.shape[0])

#     # generate 340 negative data points
#     n_P = [400, 30, 20]
#     # n_P = [800, 60, 40]
#     P = 3
#     prng = np.random.RandomState(201)

#     # the first 300 data points come from mean 2-D mvn with mean [2.2, 1.5] and next 20 come from
#     # 2-D mvn with mean [0, 3] and next 20 from 2-D mvn with mean [0, 0.5]
#     mu_PD = np.asarray([
#         [2.25, 1.5],
#         [0.0, 3.0],
#         [0.0, 0.5],
#         ])

#     cov_PDD = np.vstack([
#         np.diag([.1, .2])[np.newaxis,:],
#         np.diag([.05, .05])[np.newaxis,:],
#         np.diag([.05, .05])[np.newaxis,:],
#         ])

#     xneg_list = list()
#     for p in range(P):
#         x_ND = prng.multivariate_normal(mu_PD[p], cov_PDD[p], size=n_P[p])
#         xneg_list.append(x_ND)
#     x_neg_ND = np.vstack(xneg_list)
#     y_neg_N = np.zeros(x_neg_ND.shape[0])

#     x_ND = np.vstack([x_pos_ND, x_neg_ND])
#     y_N = np.hstack([y_pos_N, y_neg_N])

#     x_ND = (x_ND - np.mean(x_ND, axis=0))/np.std(x_ND, axis=0)

#     x_pos_ND = x_ND[y_N == 1]
#     x_neg_ND = x_ND[y_N == 0]

#     prng = np.random.RandomState(0)
#     shuffle_ids = prng.permutation(y_N.size)
#     train_x_ND = x_ND[shuffle_ids]
#     train_y_N = y_N[shuffle_ids]
    
#     train_x_ND_ = torch.from_numpy(train_x_ND)
#     train_y_N_ = torch.from_numpy(train_y_N)
    
# #     # initialize metric
# #     f2_score = Fbeta(2)
# #     f2_score.initialize()
# #     f2_score.enforce_special_case_positive()

#     # initialize metric
#     recall_gv_precision_80 = RecallGvPrecision(0.8)
#     recall_gv_precision_80.initialize()
#     recall_gv_precision_80.enforce_special_case_positive()
#     recall_gv_precision_80.set_cs_special_case_positive(True)    
    
    
#     device='cpu'
#     # create a model and criterion layer
#     torch.manual_seed(41)
#     model = Net().to(device)
# #     criterion = MetricLayer(f2_score).to(device)
#     criterion = MetricLayer(recall_gv_precision_80).to(device)
# #     criterion = nn.BCEWithLogitsLoss().to(device)

#     # forward pass
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
#     niter = 50
#     objective_list = []
#     for ii in range(0, niter):
    
#         optimizer.zero_grad()
#         output = model(train_x_ND_)
#         objective = criterion(output, train_y_N_)

#         # backward pass
#         objective.backward()
#         optimizer.step()
        
#         print("-" * 10)
#         print("loss = {}".format(objective.data))
#         print("metric = {}".format(recall_gv_precision_80.compute_metric(output.detach().numpy(), train_y_N)))
#         print("learned weights = {}".format(list(model.parameters())[0].data[0]))
#         print("learned bias = {}".format(list(model.parameters())[1].data[0]))
        
        
#         if ii>0:
#             curr_objective = objective.data.detach().numpy()
#             if abs(objective_list[-1] - curr_objective)<=0.05:
#                 print('Convergence reached.. Stopping training..')
#                 break
            
#         objective_list.append(objective.data.detach().numpy())
        