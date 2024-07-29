import numpy as np
import torch,collections
from tensorboardX import SummaryWriter
import networkx as nx
import os
from .data_obj import DataProcessor, AbnormalSubgraph
from .policy_agent import RefinementAgent
from .symbol import EXPAND, EXCLUDE, VIRTUAL_EXCLUDE_NODE
from utils import  eval_scores_timeporal,eval_scores,generate_outer_boundary_3,generate_outer_boundary_with_dynamic_threshold,eval_scores_timeporal
import scipy.stats as stats
import math

class AbnormalRefine:
    def __init__(self, args, graph, feat_mat, train_abnormalsubgraphs, val_abnormalsubgraphs, pred_abnormalsubgraphs, cost_choice, benford_G, node_chisquares,Tgraph):
        self.args = args
        self.node_chisquares = node_chisquares
        self.data_processor = DataProcessor(args, args.dataset, feat_mat, graph, train_abnormalsubgraphs, val_abnormalsubgraphs, node_chisquares,Tgraph)
        self.graph = graph
        self.cost_choice = cost_choice
        self.feat_mat = feat_mat
        self.agent = RefinementAgent(args)
        self.valid_abnormalsubgraphs, self.pred_abnormalsubgraphs = val_abnormalsubgraphs, pred_abnormalsubgraphs
        self.writer = SummaryWriter(args.writer_dir)
        self.max_step = args.max_step
        self.best_epoch = None
        self.benford_G = benford_G
        self.Tgraph = Tgraph




    def load_net(self, filename=None):
        file = self.args.writer_dir + "/subgraphr.pt" if not filename else filename
        print(f"Load net from {file} at Epoch{self.best_epoch}")
        data = torch.load(file)
        self.agent.expand_net.load_state_dict(data[EXPAND])
        self.agent.exclude_net.load_state_dict(data[EXCLUDE])

    def save_net(self, file_name=None):
        data = {
            EXPAND: self.agent.expand_net.state_dict(),
            EXCLUDE: self.agent.exclude_net.state_dict()
        }
        f_name = self.args.writer_dir + "/subgraphr.pt" if not file_name else file_name
        torch.save(data, f_name)

    def train(self):
        self.agent.exclude_net.train()
        self.agent.expand_net.train()

        gamma = self.args.gamma

        n_episode = self.args.n_episode
        n_epoch = self.args.n_epoch
        n_sample = 0

        prg_bar = range(n_epoch)
        # Validation set for getting best model
        val_data,_ = self.data_processor.generate_data(batch_size=n_episode, valid=True)

        steps, expand_steps, exclude_steps = [], [], []
        total_exclude_rewards, total_expand_rewards = [], []

        # best_f, best_j, best_nmi = 0, 0, 0
        best_density = 0
        best_fei = 0
        best_ratio = 0

        for epoch in prg_bar:
            n_sample += 1
            exclude_log_probs, exclude_rewards = [], []
            expand_log_probs, expand_rewards = [], []

            batch_data,total_nodes = self.data_processor.generate_data(batch_size=n_episode)
            # print("batch data:"+str(batch_data))

            for i in range(len(batch_data)):
                obj = batch_data[i]
                nodes = total_nodes[i]
                episode_exclude_rewards, episode_expand_rewards = [], []
                total_exclude_reward, total_expand_reward = 0, 0
                step, expand_step, exclude_step = 0, 0, 0

                expand, exclude = True, True

                while True:
                    if exclude:
                        exclude_action = self.agent.choose_action(obj, EXCLUDE)
                        if exclude_action is not None:
                            # exclude_log_probs.append(exclude_action["log_prob"])
                            # # Apply EXCLUDE
                            # for nodes in total_nodes:
                                exclude_log_probs.append(exclude_action["log_prob"])
                                exclude_reward = obj.apply_exclude(exclude_action["node"], self.cost_choice,self.graph,nodes,self.benford_G)
                                total_exclude_reward += exclude_reward
                                episode_exclude_rewards.append(exclude_reward)
                                exclude_step += 1
                        else:
                            exclude = False

                    if expand:
                        expand_action = self.agent.choose_action(obj, EXPAND)
                        if expand_action is not None:
                            # expand_log_probs.append(expand_action["log_prob"])
                            # Apply EXPAND
                            # for nodes in total_nodes:
                                expand_log_probs.append(expand_action["log_prob"])
                                expand_reward = obj.apply_expand(expand_action["node"], self.cost_choice,self.graph,nodes,self.benford_G)
                                total_expand_reward += expand_reward
                                episode_expand_rewards.append(expand_reward)
                                expand_step += 1
                        else:
                            expand = False
                    next_state = obj.step(self.graph,self.agent.gcn)

                    if (not exclude and not expand) or step >= self.max_step:
                        if len(episode_exclude_rewards) > 0:
                            r = [np.sum(episode_exclude_rewards[i] * (gamma**np.array(
                                range(i, len(episode_exclude_rewards))))) for i in range(len(episode_exclude_rewards))]
                            exclude_rewards.append(np.array(r))
                            total_exclude_rewards.append(total_exclude_reward)
                        if len(episode_expand_rewards) > 0:
                            r = [np.sum(episode_expand_rewards[i] * (gamma**np.array(
                                range(i, len(episode_expand_rewards))))) for i in range(len(episode_expand_rewards))]
                            expand_rewards.append(np.array(r))
                            total_expand_rewards.append(total_expand_reward)
                        steps.append(step)
                        exclude_steps.append(exclude_step)
                        expand_steps.append(expand_step)
                        break

                    obj.feat_mat = next_state
                    step += 1

            if len(total_exclude_rewards) > 0:
                avg_total_exclude_reward = sum(total_exclude_rewards) / len(total_exclude_rewards)
                self.writer.add_scalar(f"{self.cost_choice}-Reward/AvgExcludeReward", avg_total_exclude_reward, n_sample)
            if len(total_expand_rewards) > 0:
                avg_total_expand_reward = sum(total_expand_rewards) / len(total_expand_rewards)
                self.writer.add_scalar(f"{self.cost_choice}-Reward/AvgExpandReward", avg_total_expand_reward, n_sample)

            self.writer.add_scalar(f"{self.cost_choice}-Steps/AvgSteps", sum(steps)/len(steps), epoch)
            self.writer.add_scalar(f"{self.cost_choice}-Steps/AvgExcludeSteps", sum(exclude_steps)/len(exclude_steps), epoch)
            self.writer.add_scalar(f"{self.cost_choice}-Steps/AvgExpandSteps", sum(expand_steps)/len(expand_steps), epoch)

            if len(exclude_rewards):
                exclude_rewards = np.concatenate(exclude_rewards, axis=0)
                exclude_rewards = (exclude_rewards - np.mean(exclude_rewards)) / (np.std(exclude_rewards) + 1e-9)
                self.agent.learn(torch.stack(exclude_log_probs), torch.from_numpy(exclude_rewards), cal_type=EXCLUDE)
            if len(expand_rewards):
                expand_rewards = np.concatenate(expand_rewards, axis=0)
                expand_rewards = (expand_rewards - np.mean(expand_rewards)) / (np.std(expand_rewards)+1e-9)
                self.agent.learn(torch.stack(expand_log_probs), torch.from_numpy(expand_rewards), cal_type=EXPAND)

            # TODO: Validation on val-set and save the best model
            if (epoch + 1) % 20 == 0:
                benford = []
                for i in range(9):
                    benford.append(math.log10(1 + 1 / (i + 1)))
                pre_density = []
                pre_fei = []
                pre_ratio = []
                for i in val_data:
                    # gr = self.graph.subgraph(i)
                    gr = self.benford_G.subgraph(i)
                    t = nx.get_edge_attributes(gr, 'weight')
                    weights = 0
                    el = []
                    et = 0
                    gr = nx.convert_node_labels_to_integers(gr)
                    for e in gr.edges():
                        for w in gr[e[0]][e[1]]['weight']:
                            el.append([e[0], e[1], self.get_start_digit(w)])

                    res = []
                    for j in t:
                        res += t[j]
                    xx = collections.Counter(res)
                    obs = []
                    for i in range(1, 10):
                        if i not in xx:
                            obs.append(0)
                        else:
                            obs.append(xx[i])
                    times = sum(obs)
                    c3, p3 = stats.chisquare(obs, np.array(benford) * times)
                    for i in range(gr.number_of_nodes()):
                        tmp_deg = sum([len(gr[i][j]['weight']) for j in gr[i]])
                        et += tmp_deg
                    pre_fei.append((c3/gr.number_of_nodes()))
                    pre_ratio.append(self.calculate_ratio(c3,et,gr))

                avg_pre_fei = sum(pre_fei) / len(pre_fei)
                avg_pre_ratio = sum(pre_ratio) / len(pre_ratio)
                refine_val = self.refine_AbnormalSubgraph(valid=True, val_pred=val_data)
                new_f, new_j, new_nmi = eval_scores(self.graph, refine_val, self.valid_abnormalsubgraphs, self.benford_G, tmp_print=False)

                now_density = []
                now_fei = []
                now_ratio = []
                for i in refine_val:

                    gr = self.benford_G.subgraph(i)
                    t = nx.get_edge_attributes(gr, 'weight')
                    weights = 0
                    el = []
                    et = 0
                    gr = nx.convert_node_labels_to_integers(gr)
                    for e in gr.edges():
                        for w in gr[e[0]][e[1]]['weight']:
                            el.append([e[0], e[1], self.get_start_digit(w)])

                    res = []
                    for j in t:
                        res += t[j]
                    xx = collections.Counter(res)
                    obs = []
                    for i in range(1, 10):
                        if i not in xx:
                            obs.append(0)
                        else:
                            obs.append(xx[i])
                    times = sum(obs)
                    c3, p3 = stats.chisquare(obs, np.array(benford) * times)
                    for i in range(gr.number_of_nodes()):
                        tmp_deg = sum([len(gr[i][j]['weight']) for j in gr[i]])
                        et += tmp_deg

                    now_fei.append((c3 / gr.number_of_nodes()))
                    now_ratio.append(self.calculate_ratio(c3,et,gr))
                avg_now_fei = sum(now_fei)/ len(now_fei)
                avg_now_ratio = sum(now_ratio) / len(now_ratio)

                if avg_now_ratio - avg_pre_ratio > 0:

                    print(f"[Eval-Epoch{epoch+1}] Improve ratio {avg_now_ratio - avg_pre_ratio :.04f}, ")

                    if avg_now_ratio >= best_ratio and epoch >= 80:
                        best_ratio = avg_now_ratio
                        self.best_epoch = epoch
                        self.save_net(self.args.writer_dir + f"/subgraphr_eval_best.pt")
        # TODO: Save model
        self.save_net()

    def calculate_ratio(self,c3, et, subgraph):
        if subgraph.number_of_nodes() == 0:
            return 0

        phi = c3 / subgraph.number_of_nodes()
        density = et / (2 * subgraph.number_of_nodes())

        if math.isnan(phi) or math.isinf(phi) or math.isnan(density) or math.isinf(density):
            return 0
        return phi / density

    def node_chisquare(self,edgelistl, node_num, dist, adjust_idx=1, directed='both', diff_func='chi'):

        node_induced_dist = [[] for i in range(node_num)]
        node_chis = []
        for edge in edgelistl:
            if directed == 'both' or directed == 'out':
                node_induced_dist[edge[0]].append(edge[2])
            if directed == 'both' or directed == 'in':
                node_induced_dist[edge[1]].append(edge[2])
        for node_dist in node_induced_dist:
            count_dist = self.count_occ(node_dist, len(dist), adjust_idx)
            if diff_func == 'chi':
                node_chis.append(stats.chisquare(count_dist, sum(count_dist) * np.array(dist))[0])
            elif diff_func == 'kl':
                node_chis.append(self.kl_divergence(np.array(count_dist) / sum(count_dist), np.array(dist)))
        return node_chis

    def kl_divergence(self,p, q):
        return sum(p[i] * math.log2(p[i] / q[i]) for i in range(len(p)) if p[i] > 0)

    def count_occ(self,l, dist_len, adjust_idx=0):
        result = [0 for i in range(dist_len)]
        for e in l:
            result[e - adjust_idx] += 1
        return result

    def get_start_digit(self,v):
        if v == 0:
            return 0
        if v < 0:
            v = -v
        while v < 1:
            v = v * 10
        return int(str(v)[:1])

    def get_refine(self, filename=None):
        if filename:
            self.load_net(filename)
        elif os.path.exists(self.args.writer_dir + f"/subgraphr_eval_best.pt"):
            self.load_net(self.args.writer_dir + "/subgraphr_eval_best.pt")
        else:
            self.load_net()
        refine_abnormalsubgraphs = self.refine_AbnormalSubgraph(valid=False, val_pred=False)
        lengths = np.array([len(pred_subs) for pred_subs in refine_abnormalsubgraphs])
        print(f"[Refine] Pred size {len(refine_abnormalsubgraphs)}, Avg Length {np.mean(lengths):.04f}")
        return refine_abnormalsubgraphs


    def refine_AbnormalSubgraph(self, valid=False, val_pred=None):
        new_preds = []
        pred_subgraphs = self.pred_abnormalsubgraphs if not valid else val_pred
        benford = []
        xs = [i for i in range(1, 10)]
        for i in range(9):
            benford.append(math.log10(1 + 1 / (i + 1)))

        for i in range(len(pred_subgraphs)):
            pred = pred_subgraphs[i]
            pred = sorted(pred)

            predsub=self.benford_G.subgraph(pred)
            respred = []
            t = nx.get_edge_attributes(predsub, 'weight')
            for j in t:
                respred += t[j]
            x = collections.Counter(respred)
            obs = []
            for i in range(1, 10):
                if i not in x:
                    obs.append(0)
                else:
                    obs.append(x[i])
            times = sum(obs)
            c3, p3 = stats.chisquare(obs, np.array(benford) * times)
            predchi = 0
            for i in range(9):
                denominator = benford[i] * times
                if denominator != 0:
                    predchi += (obs[i] - denominator) ** 2 / denominator
                else:
                    predchi += 0
            predphi = c3 / predsub.number_of_nodes()

            # outer_bound = generate_outer_boundary_3(self.graph, pred, max_size=200)
            outer_bound = generate_outer_boundary_with_dynamic_threshold(self.graph, pred, self.Tgraph,max_size=200)
            nodes = sorted(pred + outer_bound)

            expand, exclude = True, True
            mapping = {idx: node for idx, node in enumerate(nodes)}

            sub_obj = AbnormalSubgraph(self.feat_mat[nodes, :], pred, None, nodes, self.graph.subgraph(nodes), mapping,
                                expand=expand)

            step = 0

            while True:
                step += 1

                if exclude:
                    exclude_action = self.agent.choose_action(sub_obj, "exclude")
                    if exclude_action is not None:
                        node = exclude_action["node"]
                        sub_ori = []
                        sub_ori = sub_obj.pred_subgraphs.copy()
                        if node in sub_ori:
                            sub_ori.remove(node)
                            sub = self.benford_G.subgraph(sub_ori)
                            pre = sub_obj.pred_subgraphs
                            if len(sub) > 0:
                                if nx.is_connected(sub):
                                    res = []
                                    t = nx.get_edge_attributes(sub, 'weight')
                                    for j in t:
                                        res += t[j]
                                    x = collections.Counter(res)
                                    obs = []
                                    for i in range(1, 10):
                                        if i not in x:
                                            obs.append(0)
                                        else:
                                            obs.append(x[i])
                                    times = sum(obs)
                                    c3, p3 = stats.chisquare(obs, np.array(benford) * times)
                                    chi = 0
                                    for i in range(9):
                                        denominator = benford[i] * times
                                        if denominator != 0:
                                            chi += (obs[i] - denominator) ** 2 / denominator
                                        else:

                                            chi += 0
                                    phi = c3 / sub.number_of_nodes()
                                    if (not math.isnan(phi)) and phi > predphi and chi >predchi:
                                        sub_obj.pred_subgraphs.remove(node)

                            else:
                                sub_ori.append(node)

                    else:
                        exclude = False
                if expand:
                    expand_action = self.agent.choose_action(sub_obj, "expand")
                    if expand_action is not None:
                        node = expand_action["node"]
                        sub_ori = sub_obj.pred_subgraphs.copy()
                        if node in sub_ori:
                            sub_ori.append(node)
                            sub = self.benford_G.subgraph(sub_ori)
                            pre = sub_obj.pred_subgraphs
                            if len(sub) > 0:
                                if nx.is_connected(sub):
                                    res = []
                                    t = nx.get_edge_attributes(sub, 'weight')
                                    for j in t:
                                        res += t[j]
                                    x = collections.Counter(res)
                                    obs = []
                                    for i in range(1, 10):
                                        if i not in x:
                                            obs.append(0)
                                        else:
                                            obs.append(x[i])
                                    times = sum(obs)
                                    c3, p3 = stats.chisquare(obs, np.array(benford) * times)
                                    chi = 0
                                    for i in range(9):
                                        denominator = benford[i] * times
                                        if denominator != 0:
                                            chi += (obs[i] - denominator) ** 2 / denominator
                                        else:
                                            chi += 0
                                    phi = c3 / sub.number_of_nodes()
                                    if (not math.isnan(phi)) and phi > predphi and chi> predchi:
                                        sub_obj.pred_subgraphs.append(node)

                            else:
                                sub_ori.remove(node)
                    else:
                        expand = False
                if (not exclude and not expand):
                    break
                next_state = sub_obj.step(self.graph,self.agent.gcn)
                sub_obj.feat_mat = next_state

            if VIRTUAL_EXCLUDE_NODE in sub_obj.pred_subgraphs:
                sub_obj.pred_subgraphs.remove(VIRTUAL_EXCLUDE_NODE)
            if len(sub_obj.pred_subgraphs) > 0:
                new_preds.append(sub_obj.pred_subgraphs)

        return new_preds
