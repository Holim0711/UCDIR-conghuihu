import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
from torch.autograd import Function

### https://github.com/fungtion/DANN/blob/master/models/functions.py
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

### https://amaarora.github.io/2020/08/30/gempool.html#pytorch-implementation
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class method2_encoder(nn.Module):
    def __init__(self, smg, base_encoder, dim=128, total_dim=1536, norm_layer=None):
        super(method2_encoder, self).__init__()

        self.smg = smg
        if norm_layer:
            self.base_encoder = base_encoder(num_classes=dim, norm_layer=norm_layer)
        else:
            self.base_encoder = base_encoder(num_classes=dim)

        self.len_smg = len(self.smg)
        self.total_dim = total_dim
        self.each_dim = self.total_dim // self.len_smg

        ### remove downsampling
        self.base_encoder.layer3[0].conv2 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,groups=1,bias=False,dilation=1)
        self.base_encoder.layer3[0].downsample[0] = nn.Conv2d(512,1024,kernel_size=1,stride=1,bias=False)

        self.base_encoder.avgpool = nn.Identity()
        self.base_encoder.fc = nn.Unflatten(1, (2048, 14, 14))

        if 's' in self.smg:
            self.avgpool_fc = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(1), 
                nn.Linear(2048, self.each_dim)
            )
        if 'm' in self.smg:
            self.maxpool_fc = nn.Sequential(
                nn.AdaptiveMaxPool2d((1,1)),
                nn.Flatten(1), 
                nn.Linear(2048, self.each_dim)
            )
        if 'g' in self.smg:
            self.gempool_fc = nn.Sequential(
                GeM(),
                nn.Flatten(1), 
                nn.Linear(2048, self.each_dim)
            )

        self.final_fc = nn.Linear(self.total_dim, dim)

    def forward(self, x):
        x = self.base_encoder(x)
        x1 = None
        x2 = None
        x3 = None
        
        if 's' in self.smg:
            x1 = self.avgpool_fc(x)
            x1 = F.normalize(x1, dim=1) 
        if 'm' in self.smg:
            x2 = self.maxpool_fc(x)
            x2 = F.normalize(x2, dim=1)
        if 'g' in self.smg:
            x3 = self.gempool_fc(x)
            x3 = F.normalize(x3, dim=1)

        if self.len_smg == 3:
            x = torch.cat([x1,x2,x3], dim=1)
            x = F.normalize(x, dim=1)
        elif self.len_smg == 2:
            if x1 is None: x = torch.cat([x2,x3], dim=1)
            elif x2 is None: x = torch.cat([x1,x3], dim=1)
            elif x3 is None: x = torch.cat([x1,x2], dim=1)
            x = F.normalize(x, dim=1)
        elif self.len_smg == 1:
            if x1 is not None: x = x1
            elif x2 is not None: x = x2
            elif x3 is not None: x = x3
        x = self.final_fc(x)
        return x

class method3_encoder(nn.Module):
    def __init__(self, base_encoder, dim=128, norm_layer=None):
        super(method3_encoder, self).__init__()

        if norm_layer:
            self.featurizer = base_encoder(num_classes=dim, norm_layer=norm_layer)
        else:
            self.featurizer = base_encoder(num_classes=dim)

        self.featurizer.fc = nn.Identity()
        self.final_fc = nn.Linear(2048, dim)
        self.domain_classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, alpha):
        feature = self.featurizer(x)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        encoder_output = self.final_fc(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return encoder_output, domain_output

class UCDIR(nn.Module):

    def __init__(self, base_encoder, dim=128, K_A=65536, K_B=65536,
                 m=0.999, T=0.1, mlp=False, selfentro_temp=0.2,
                 num_cluster=None, cwcon_filterthresh=0.2, method='default',
                 smg=None):

        super(UCDIR, self).__init__()

        self.K_A = K_A
        self.K_B = K_B
        self.m = m
        self.T = T

        self.selfentro_temp = selfentro_temp
        self.num_cluster = num_cluster
        self.cwcon_filterthresh = cwcon_filterthresh

        norm_layer = partial(SplitBatchNorm, num_splits=2)

        self.method = method

        if self.method == 'default':
            self.encoder_q = base_encoder(num_classes=dim)
            self.encoder_k = base_encoder(num_classes=dim, norm_layer=norm_layer)
            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        elif self.method == 'method1':
            self.encoder_q_A = base_encoder(num_classes=dim)
            self.encoder_k_A = base_encoder(num_classes=dim, norm_layer=norm_layer)
            self.encoder_q_B = base_encoder(num_classes=dim)
            self.encoder_k_B = base_encoder(num_classes=dim, norm_layer=norm_layer)

            if mlp:  # hack: brute-force replacement
                dim_mlpA = self.encoder_q_A.fc.weight.shape[1]
                self.encoder_q_A.fc = nn.Sequential(nn.Linear(dim_mlpA, dim_mlpA), nn.ReLU(), self.encoder_q_A.fc)
                self.encoder_k_A.fc = nn.Sequential(nn.Linear(dim_mlpA, dim_mlpA), nn.ReLU(), self.encoder_k_A.fc)
            
                dim_mlpB = self.encoder_q_B.fc.weight.shape[1]
                self.encoder_q_B.fc = nn.Sequential(nn.Linear(dim_mlpB, dim_mlpB), nn.ReLU(), self.encoder_q_B.fc)
                self.encoder_k_B.fc = nn.Sequential(nn.Linear(dim_mlpB, dim_mlpB), nn.ReLU(), self.encoder_k_B.fc)

            for param_q, param_k in zip(self.encoder_q_A.parameters(), self.encoder_k_A.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
        
            for param_q, param_k in zip(self.encoder_q_B.parameters(), self.encoder_k_B.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        elif self.method == 'method2':
            self.smg = smg
            if self.smg is None:
                print('method 2 needs args.smg')
                exit()

            self.encoder_q = method2_encoder(self.smg, base_encoder, dim=128, total_dim=1536)
            self.encoder_k = method2_encoder(self.smg, base_encoder, dim=128, total_dim=1536, norm_layer=norm_layer)

            if mlp:  # hack: brute-force replacement
                self.encoder_q.final_fc = nn.Sequential(nn.Linear(1536, 1536), nn.ReLU(), self.encoder_q.final_fc)
                self.encoder_k.final_fc = nn.Sequential(nn.Linear(1536, 1536), nn.ReLU(), self.encoder_k.final_fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        elif self.method == 'method3':
            self.encoder_q = method3_encoder(base_encoder, dim=128)
            self.encoder_k = method3_encoder(base_encoder, dim=128, norm_layer=norm_layer)
            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.final_fc.weight.shape[1]
                self.encoder_q.final_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.final_fc)
                self.encoder_k.final_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.final_fc)
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        # create the queues
        self.register_buffer("queue_A", torch.randn(dim, K_A))
        self.queue_A = F.normalize(self.queue_A, dim=0)
        self.register_buffer("queue_A_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_B", torch.randn(dim, K_B))
        self.queue_B = F.normalize(self.queue_B, dim=0)
        self.register_buffer("queue_B_ptr", torch.zeros(1, dtype=torch.long))

        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        if self.method == 'default':
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        elif self.method == 'method1':
            for param_q, param_k in zip(self.encoder_q_A.parameters(), self.encoder_k_A.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
            for param_q, param_k in zip(self.encoder_q_B.parameters(), self.encoder_k_B.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        elif self.method == 'method2':
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        elif self.method == 'method3':
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_singlegpu(self, keys, key_ids, domain_id):

        if domain_id == 'A':
            self.queue_A.index_copy_(1, key_ids, keys.T)
        elif domain_id == 'B':
            self.queue_B.index_copy_(1, key_ids, keys.T)

    @torch.no_grad()
    def _batch_shuffle_singlegpu(self, x):

        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_singlegpu(self, x, idx_unshuffle):

        return x[idx_unshuffle]

    def forward(self, im_q_A, im_q_B, im_k_A=None, im_id_A=None,
                im_k_B=None, im_id_B=None, is_eval=False,
                cluster_result=None, criterion=None,
                alpha=None, criterion_domain=None):
        
        if self.method == 'default':
            im_q = torch.cat([im_q_A, im_q_B], dim=0)
            if is_eval:
                k = self.encoder_k(im_q)
                k = F.normalize(k, dim=1)
                k_A, k_B = torch.split(k, im_q_A.shape[0])
                return k_A, k_B
            q = self.encoder_q(im_q)
            q = F.normalize(q, dim=1)
            q_A, q_B = torch.split(q, im_q_A.shape[0])
            im_k = torch.cat([im_k_A, im_k_B], dim=0)
            with torch.no_grad():
                self._momentum_update_key_encoder()
                im_k, idx_unshuffle = self._batch_shuffle_singlegpu(im_k)
                k = self.encoder_k(im_k)
                k = F.normalize(k, dim=1)
                k = self._batch_unshuffle_singlegpu(k, idx_unshuffle)
                k_A, k_B = torch.split(k, im_k_A.shape[0])
        elif self.method == 'method1':
            if is_eval:
                k_A = self.encoder_k_A(im_q_A)
                k_B = self.encoder_k_B(im_q_B)
                k_A = F.normalize(k_A, dim=1)
                k_B = F.normalize(k_B, dim=1)
                return k_A, k_B
            q_A = self.encoder_q_A(im_q_A)
            q_A = F.normalize(q_A, dim=1)    
            q_B = self.encoder_q_B(im_q_B)
            q_B = F.normalize(q_B, dim=1)    
            with torch.no_grad():
                self._momentum_update_key_encoder()
                im_k_A, idx_unshuffle_A = self._batch_shuffle_singlegpu(im_k_A)
                im_k_B, idx_unshuffle_B = self._batch_shuffle_singlegpu(im_k_B)
                k_A = self.encoder_k_A(im_k_A)
                k_A = F.normalize(k_A, dim=1)
                k_B = self.encoder_k_B(im_k_B)
                k_B = F.normalize(k_B, dim=1)
                k_A = self._batch_unshuffle_singlegpu(k_A, idx_unshuffle_A)
                k_B = self._batch_unshuffle_singlegpu(k_B, idx_unshuffle_B)
        elif self.method == 'method2':
            im_q = torch.cat([im_q_A, im_q_B], dim=0)
            if is_eval:
                k = self.encoder_k(im_q)
                k = F.normalize(k, dim=1)
                k_A, k_B = torch.split(k, im_q_A.shape[0])
                return k_A, k_B
            q = self.encoder_q(im_q)
            q = F.normalize(q, dim=1)
            q_A, q_B = torch.split(q, im_q_A.shape[0])
            im_k = torch.cat([im_k_A, im_k_B], dim=0)
            with torch.no_grad():
                self._momentum_update_key_encoder()
                im_k, idx_unshuffle = self._batch_shuffle_singlegpu(im_k)
                k = self.encoder_k(im_k)
                k = F.normalize(k, dim=1)
                k = self._batch_unshuffle_singlegpu(k, idx_unshuffle)
                k_A, k_B = torch.split(k, im_k_A.shape[0])
        elif self.method == 'method3':
            im_q = torch.cat([im_q_A, im_q_B], dim=0)
            if is_eval:
                k = self.encoder_k.featurizer(im_q)
                k = self.encoder_k.final_fc(k)
                k = F.normalize(k, dim=1)
                k_A, k_B = torch.split(k, im_q_A.shape[0])
                return k_A, k_B
            
            if alpha is None:
                print("in method3, train mode need alpha value")
                exit()
            if criterion_domain is None:
                print("in method3, train mode need criterion_domain")
                exit()

            q, d = self.encoder_q(im_q, alpha)
            q = F.normalize(q, dim=1)
            q_A, q_B = torch.split(q, im_q_A.shape[0])
            loss_domain_A, loss_domain_B = self.domain_cls_loss(d, criterion_domain)
            losses_domain = {'domain_loss_A': loss_domain_A, 'domain_loss_B' : loss_domain_B }
            im_k = torch.cat([im_k_A, im_k_B], dim=0)
            with torch.no_grad():
                self._momentum_update_key_encoder()
                im_k, idx_unshuffle = self._batch_shuffle_singlegpu(im_k)
                if self.method == 'method3':
                    k = self.encoder_k.featurizer(im_k)
                    k = self.encoder_k.final_fc(k)
                else:
                    k = self.encoder_k(im_k)
                k = F.normalize(k, dim=1)
                k = self._batch_unshuffle_singlegpu(k, idx_unshuffle)
                k_A, k_B = torch.split(k, im_k_A.shape[0])


        self._dequeue_and_enqueue_singlegpu(k_A, im_id_A, 'A')
        self._dequeue_and_enqueue_singlegpu(k_B, im_id_B, 'B')
        loss_instcon_A, \
        loss_instcon_B = self.instance_contrastive_loss(q_A, k_A, im_id_A,
                                                        q_B, k_B, im_id_B,
                                                        criterion)
        losses_instcon = {'domain_A': loss_instcon_A,
                             'domain_B': loss_instcon_B}
        if cluster_result is not None:
            loss_cwcon_A, \
            loss_cwcon_B = self.cluster_contrastive_loss(q_A, k_A, im_id_A,
                                                         q_B, k_B, im_id_B,
                                                         cluster_result)
            losses_cwcon = {'domain_A': loss_cwcon_A,
                        'domain_B': loss_cwcon_B}
            losses_selfentro = self.self_entropy_loss(q_A, q_B, cluster_result)
            losses_distlogit = self.dist_of_logit_loss(q_A, q_B, cluster_result, self.num_cluster)
            
            if self.method == 'method3':
                return losses_instcon, q_A, q_B, losses_selfentro, losses_distlogit, losses_cwcon, losses_domain
            else:
                return losses_instcon, q_A, q_B, losses_selfentro, losses_distlogit, losses_cwcon
        else:
            if self.method == 'method3':
                return losses_instcon, None, None, None, None, None, losses_domain
            else:
                return losses_instcon, None, None, None, None, None

    def domain_cls_loss(self, domain_cls_res, criterion_domain):
        domain_A_label = torch.zeros(domain_cls_res.shape[0]//2, dtype=torch.long).cuda()
        domain_B_label = torch.ones(domain_cls_res.shape[0]//2, dtype=torch.long).cuda()

        d_res_A, d_res_B = torch.split(domain_cls_res, domain_cls_res.shape[0] // 2)

        domain_loss_A = criterion_domain(d_res_A, domain_A_label)
        domain_loss_B = criterion_domain(d_res_B, domain_B_label)
        return domain_loss_A, domain_loss_B

    def instance_contrastive_loss(self,
                                  q_A, k_A, im_id_A,
                                  q_B, k_B, im_id_B,
                                  criterion):

        l_pos_A = torch.einsum('nc,nc->n', [q_A, k_A]).unsqueeze(-1)
        l_pos_B = torch.einsum('nc,nc->n', [q_B, k_B]).unsqueeze(-1)

        l_all_A = torch.matmul(q_A, self.queue_A.clone().detach())
        l_all_B = torch.matmul(q_B, self.queue_B.clone().detach())

        mask_A = torch.arange(self.queue_A.shape[1]).cuda() != im_id_A[:, None]
        l_neg_A = torch.masked_select(l_all_A, mask_A).reshape(q_A.shape[0], -1)

        mask_B = torch.arange(self.queue_B.shape[1]).cuda() != im_id_B[:, None]
        l_neg_B = torch.masked_select(l_all_B, mask_B).reshape(q_B.shape[0], -1)

        logits_A = torch.cat([l_pos_A, l_neg_A], dim=1)
        logits_B = torch.cat([l_pos_B, l_neg_B], dim=1)

        logits_A /= self.T
        logits_B /= self.T

        labels_A = torch.zeros(logits_A.shape[0], dtype=torch.long).cuda()
        labels_B = torch.zeros(logits_B.shape[0], dtype=torch.long).cuda()

        loss_A = criterion(logits_A, labels_A)
        loss_B = criterion(logits_B, labels_B)


        return loss_A, loss_B

    def cluster_contrastive_loss(self, q_A, k_A, im_id_A, q_B, k_B, im_id_B, cluster_result):

        all_losses = {'domain_A': [], 'domain_B': []}

        for domain_id in ['A', 'B']:
            if domain_id == 'A':
                im_id = im_id_A
                q_feat = q_A
                k_feat = k_A
                queue = self.queue_A.clone().detach()
            else:
                im_id = im_id_B
                q_feat = q_B
                k_feat = k_B
                queue = self.queue_B.clone().detach()

            mask = 1.0
            for n, (im2cluster, prototypes) in enumerate(zip(cluster_result['im2cluster_' + domain_id],
                                                             cluster_result['centroids_' + domain_id])):

                cor_cluster_id = im2cluster[im_id]

                mask *= torch.eq(cor_cluster_id.contiguous().view(-1, 1),
                                 im2cluster.contiguous().view(1, -1)).float()  # batch size x queue length

                all_score = torch.div(torch.matmul(q_feat, queue), self.T)
                
                exp_all_score = torch.exp(all_score)

                log_prob = all_score - torch.log(exp_all_score.sum(1, keepdim=True))

                mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

                cor_proto = prototypes[cor_cluster_id]
                inst_pos_value = torch.exp(
                    torch.div(torch.einsum('nc,nc->n', [k_feat, cor_proto]), self.T))  # N
                inst_all_value = torch.exp(
                    torch.div(torch.einsum('nc,ck->nk', [k_feat, prototypes.T]), self.T))  # N x r
                
                filters = ((inst_pos_value / torch.sum(inst_all_value, dim=1)) > self.cwcon_filterthresh).float()
                filters_sum = filters.sum()

                loss = - (filters * mean_log_prob_pos).sum() / (filters_sum + 1e-8)
                all_losses['domain_' + domain_id].append(loss)

        return torch.mean(torch.stack(all_losses['domain_A'])), torch.mean(torch.stack(all_losses['domain_B']))

    def self_entropy_loss(self, q_A, q_B, cluster_result):

        losses_selfentro = {}
        for feat_domain in ['A', 'B']:
            if feat_domain == 'A':
                feat = q_A
            else:
                feat = q_B

            cross_proto_domains = ['A', 'B']
            for cross_proto_domain in cross_proto_domains:
                for n, (im2cluster, self_proto, cross_proto) in enumerate(
                        zip(cluster_result['im2cluster_' + feat_domain],
                            cluster_result['centroids_' + feat_domain],
                            cluster_result['centroids_' + cross_proto_domain])):

                    if str(self_proto.shape[0]) in self.num_cluster:

                        key_selfentro = 'feat_domain_' + feat_domain + '-proto_domain_' \
                                        + cross_proto_domain + '-cluster_' + str(cross_proto.shape[0])
                        if key_selfentro in losses_selfentro.keys():
                            losses_selfentro[key_selfentro].append(self.self_entropy_loss_onepair(feat, cross_proto))
                        else:
                            losses_selfentro[key_selfentro] = [self.self_entropy_loss_onepair(feat, cross_proto)]
        return losses_selfentro

    def self_entropy_loss_onepair(self, feat, prototype):

        logits = torch.div(torch.matmul(feat, prototype.T), self.selfentro_temp)

        self_entropy = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * F.softmax(logits, dim=1), dim=1))

        return self_entropy

    def dist_of_logit_loss(self, q_A, q_B, cluster_result, num_cluster):

        all_losses = {}

        for n, (proto_A, proto_B) in enumerate(zip(cluster_result['centroids_A'],
                                                   cluster_result['centroids_B'])):

            if str(proto_A.shape[0]) in num_cluster:
                domain_ids = ['A', 'B']

                for domain_id in domain_ids:
                    if domain_id == 'A':
                        feat = q_A
                    elif domain_id == 'B':
                        feat = q_B
                    else:
                        feat = torch.cat([q_A, q_B], dim=0)

                    loss_A_B = self.dist_of_dist_loss_onepair(feat, proto_A, proto_B)

                    key_A_B = 'feat_domain_' + domain_id + '_A_B' + '-cluster_' + str(proto_A.shape[0])
                    if key_A_B in all_losses.keys():
                        all_losses[key_A_B].append(loss_A_B.mean())
                    else:
                        all_losses[key_A_B] = [loss_A_B.mean()]

        return all_losses

    def dist_of_dist_loss_onepair(self, feat, proto_1, proto_2):

        proto1_distlogits = self.dist_cal(feat, proto_1)
        proto2_distlogits = self.dist_cal(feat, proto_2)

        loss_A_B = F.pairwise_distance(proto1_distlogits, proto2_distlogits, p=2) ** 2

        return loss_A_B

    def dist_cal(self, feat, proto, temp=0.01):

        proto_logits = F.softmax(torch.matmul(feat, proto.T) / temp, dim=1)

        proto_distlogits = 1.0 - torch.matmul(F.normalize(proto_logits, dim=1), F.normalize(proto_logits.T, dim=0))

        return proto_distlogits


# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape

        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)

            outcome = F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)
