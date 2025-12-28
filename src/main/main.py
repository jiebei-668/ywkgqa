import argparse
from sentence_transformers import SentenceTransformer, util
# 假设这些模块在你的环境中可用
from kgdao import KGDao
from llm import ChatModel
import utils  # 假设你的自定义函数在这里

if __name__ == '__main__':
    # --- 1. 环境与设置 ---

    # 模拟 args 类，因为 utils.relation_search_prune 需要它
    class Args:
        def __init__(self):
            self.remove_unnecessary_rel = True
            self.temperature_exploration = 0.3 # 示例值
    args = Args()

    # 数据库与模型设置
    neo4j_url = "neo4j://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "KOBEforever668!"
    neo4j_database = "ywkgqa"
    dao = KGDao(neo4j_url, neo4j_username, neo4j_password, neo4j_database)

    provider = "ollama"
    base_url = "http://localhost:11434"
    model_name = "qwen3:8b"
    chat_model = ChatModel(provider=provider, model_name=model_name, base_url=base_url)

    model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')

    # --- 2. 数据初始化 ---

    question = "What entity is reached from e1 via r2 and then r4?"

    # 初始主题实体 (占位符 - 请替换为通过 NER/EL 找到的实际起始节点)
    # 示例: 如果 e1 对应的 ID 是 "001"
    eids = ["e1"]
    enames = ["e1"]

    # 创建映射字典
    eid2name = dict(zip(eids, enames))
    name2eid = dict(zip(enames, eids))

    # 初始化状态变量
    pre_relations = []
    # -1 表示初始步骤没有前驱头节点状态
    pre_heads = [-1] * len(eids)
    cluster_chain_of_entities = []

    print(f"开始 QA 循环，问题: {question}")
    print(f"初始实体: {eid2name}")

    # --- 3. 推理循环 ---

    for depth in range(1, 6):
        print(f"\n=== 跳数深度 {depth} ===")

        # --- 步骤 A: 关系搜索 (批量处理) ---
        current_entity_relations_list = []

        for i, eid in enumerate(eids):
            # 获取当前实体的候选关系
            # 参数: (dao, entity_id, entity_name, history_relations, previous_head_status, question, llm, args)
            results = utils.relation_search_prune(
                dao,
                eid,
                eid2name[eid],
                pre_relations,
                pre_heads[i],
                question,
                chat_model,
                args
            )
            current_entity_relations_list.extend(results)

        print(f"找到的关系数量: {len(current_entity_relations_list)}")

        # --- 步骤 B: 字典构建 (逻辑模拟) ---
        ent_rel_ent_dict = {}

        for ent_rel in current_entity_relations_list:
            # 结构: {'entity': id, 'relation': rel_name, 'head': bool}
            curr_entity_id = ent_rel['entity']
            relation = ent_rel['relation']
            is_head = ent_rel['head']

            # 确定方向
            direction = 'head' if is_head else 'tail'

            # 搜索相邻实体
            # 参数: (dao, current_id, relation, is_head_node)
            found_entities = utils.entity_search(dao, curr_entity_id, relation, is_head)

            # 提取 ID 用于字典并更新映射
            found_ids = []
            for item in found_entities:
                f_id = item['id']
                f_name = item['name']

                # 更新全局映射
                eid2name[f_id] = f_name
                name2eid[f_name] = f_id

                found_ids.append(f_id)

            # 填充结构并进行嵌套初始化
            if curr_entity_id not in ent_rel_ent_dict:
                ent_rel_ent_dict[curr_entity_id] = {}

            if direction not in ent_rel_ent_dict[curr_entity_id]:
                ent_rel_ent_dict[curr_entity_id][direction] = {}

            ent_rel_ent_dict[curr_entity_id][direction][relation] = found_ids

        # --- 步骤 C: 剪枝与推理 ---
        # 参数: (question, graph_dict, id_map, name_map, llm, embedding_model)
        flag, chain_of_entities, entities_id, filter_relations, filter_head, new_ent_rel_ent_dict = utils.entity_condition_prune(
            question,
            ent_rel_ent_dict,
            eid2name,
            name2eid,
            chat_model,
            model
        )

        # --- 步骤 D: 状态更新 (关键) ---
        cluster_chain_of_entities.append(chain_of_entities)

        # 覆盖变量以用于下一次迭代
        eids = entities_id
        pre_relations = filter_relations
        pre_heads = filter_head

        print(f"深度 {depth} 结果 - 继续: {flag}, 实体数量: {len(eids)}")

        if not flag:
            print("推理循环因剪枝条件终止。")
            break

    print("\n=== 最终链条 ===")
    for chain in cluster_chain_of_entities:
        print(chain)
