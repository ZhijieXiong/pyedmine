import heapq
import numpy as np
from collections import defaultdict, deque

from edmine.model.LearningPathRecommendationAgent import LPRAgent


class AStarRecConceptAgent(LPRAgent):
    def __init__(self, params, objects):
        super().__init__(params, objects)
        self.rev_graph = None
        self.shortest_path2goal = None
    
    def judge_done(self, memory, master_th=0.6):
        if memory.achieve_single_goal(master_th):
            return True
        evaluator_config = self.params["evaluator_config"]
        agent_name = evaluator_config["agent_name"]
        max_concept_attempt = int(agent_name.split("-")[1])
        max_attempt_per_concept = int(agent_name.split("-")[2])
        
        # 因为memory的update机制会自动将连续两个相同知识点合并，所以需要单独检查一下习题推荐数量
        max_question_attempt = max_concept_attempt * max_attempt_per_concept
        num_question_his = 0
        for qs in memory.question_rec_history:
            num_question_his += len(qs)
        if num_question_his >= max_question_attempt:
            return True
        
        if len(memory.concept_rec_history) > max_concept_attempt:
            return True
        
        if len(memory.concept_rec_history) == max_concept_attempt:
            cur_stage = len(memory.concept_rec_history) - 1
            last_stage_qs = memory.question_rec_history[cur_stage]
            return len(last_stage_qs) >= max_attempt_per_concept
        else:
            return False
        
    def recommend_qc(self, memory, master_th=0.6, epsilon=0):
        evaluator_config = self.params["evaluator_config"]
        agent_name = evaluator_config["agent_name"]
        max_attempt_per_concept = int(agent_name.split("-")[2])
        random_generator = self.objects["random_generator"]
        
        state = memory.state_history[-1]
        
        if len(memory.concept_rec_history) == 0:
            c_id2rec = self.rec_concept(memory, master_th)
        else:
            last_stage_rec_c = memory.concept_rec_history[-1]
            cur_stage = len(memory.concept_rec_history) - 1
            last_stage_qs = memory.question_rec_history[cur_stage]
            if (state[last_stage_rec_c] >= master_th) or (len(last_stage_qs) >= max_attempt_per_concept):
                c_id2rec = self.rec_concept(memory, master_th)
            else:
                c_id2rec = last_stage_rec_c
        
        c2q = self.objects["dataset"]["c2q"]
        q_id2rec = int(random_generator.choice(c2q[c_id2rec]))
        
        return c_id2rec, q_id2rec
    
    def prepare4a_star(self):
        pre_edges = self.objects["graph"]["pre_relation_edges"]
        self.rev_graph = build_reverse_graph(pre_edges)
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        self.shortest_path2goal = {
            c_id: shortest_distance_to_goal(c_id, self.rev_graph, num_concept) 
            for c_id in range(num_concept)
        }
    
    def a_star_knowledge_path(self, state, learning_goal):
        if self.rev_graph is None or self.shortest_path2goal is None:
            self.prepare4a_star()
            
        num_nodes = self.objects["dataset"]["q_table"].shape[1]
        master_th = self.params["evaluator_config"]["master_threshold"]
        rev_graph = self.rev_graph
        h_values = self.shortest_path2goal[learning_goal]

        # 判断哪些知识点已掌握（可视为起点），可能为空，这种比较好，因为如果降低threshold，会导致前置知识点学习不到位
        mastered = set(i for i, v in enumerate(state) if v >= master_th)
        # 尽可能使mastered不为空
        # mastered = get_mastered(state, master_th)

        # 特殊情况：目标已掌握
        if state[learning_goal] >= master_th:
            return [learning_goal]

        # 启发函数 h(n) = 预计算距离
        def heuristic(n):
            return h_values[n]

        # 判断节点n是否可学习（即所有前置都掌握）
        def can_learn(n, learned_set):
            return all(pre in learned_set for pre in rev_graph[n])

        # 初始化 open list (优先队列): 存 (f, g, current_node, learned_path, learned_set)
        open_list = []
        # 初始状态：学习路径空，已掌握知识点是初始集合
        # 这里我们把所有已掌握节点作为“学习路径开始点”的参考，但真正路径从空开始，扩展学习新知识点
        # 所以起点不固定在某节点，而是由状态决定
        start_state = frozenset(mastered)
        heapq.heappush(open_list, (heuristic(learning_goal), 0, None, [], start_state))

        # 用一个集合记录访问过的状态，避免重复搜索
        visited = set()
        visited.add(start_state)

        while open_list:
            f, g, current, path, learned = heapq.heappop(open_list)

            # 如果目标已掌握，返回路径
            if learning_goal in learned:
                return path

            # 找所有可以学习的新知识点
            candidates = [n for n in range(num_nodes) if n not in learned and can_learn(n, learned)]

            for nxt in candidates:
                new_learned = set(learned)
                new_learned.add(nxt)
                new_learned_frozen = frozenset(new_learned)
                if new_learned_frozen in visited:
                    continue
                visited.add(new_learned_frozen)
                new_g = g + 1  # 学一个新知识点代价 1
                new_f = new_g + heuristic(nxt)
                new_path = path + [nxt]
                heapq.heappush(open_list, (new_f, new_g, nxt, new_path, new_learned_frozen))

        # 无法达到目标
        return None

    def a_star_rec_concept(self, memory):
        state = np.array([float(s) for s in memory.state_history[-1]])
        learning_goal = memory.learning_goals[0]
        leaning_path = self.a_star_knowledge_path(state, learning_goal)
        if leaning_path is None:
            return learning_goal
        else:
            return leaning_path[0]
        
    def rec_concept(self, memory, master_th):
        c_id2rec = self.a_star_rec_concept(memory)
        
        return int(c_id2rec)


def build_reverse_graph(edges):
    # 构建反向图（后继指向前驱）
    rev_graph = defaultdict(list)
    for pre, post in edges:
        rev_graph[post].append(pre)
    return rev_graph


def shortest_distance_to_goal(goal, graph, num_nodes):
    """
    预先用 BFS 计算从任意点到目标的最短距离（用于启发式函数h）
    无法到达目标返回一个大数（例如num_nodes+10）
    """
    dist = [num_nodes + 10] * num_nodes
    dist[goal] = 0
    queue = deque([goal])
    while queue:
        node = queue.popleft()
        for pre in graph[node]:
            if dist[pre] > dist[node] + 1:
                dist[pre] = dist[node] + 1
                queue.append(pre)
    return dist


def get_mastered(state, initial_threshold, min_threshold=0.0, step=0.05):
    threshold = initial_threshold
    while threshold >= min_threshold:
        mastered = {i for i, v in enumerate(state) if v >= threshold}
        if mastered:
            return mastered
        threshold -= step
    return set()
