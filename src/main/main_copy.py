import argparse
import json
import os
import time
from typing import List, Dict, Any, Tuple

from sentence_transformers import SentenceTransformer
from kgdao import KGDao
from llm import ChatModel
import utils


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Knowledge Graph Question Answering with Graph Reasoning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据库配置
    parser.add_argument("--neo4j_url", type=str, default="neo4j://localhost:7687",
                        help="Neo4j connection URL")
    parser.add_argument("--neo4j_username", type=str, default="neo4j",
                        help="Neo4j username")
    parser.add_argument("--neo4j_password", type=str, default="jbh966225",
                        help="Neo4j password")
    parser.add_argument("--neo4j_database", type=str, default="neo4j",
                        help="Neo4j database name")

    # LLM配置
    parser.add_argument("--llm_provider", type=str, default="ollama",
                        choices=["ollama", "openai", "custom"],
                        help="LLM provider")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:11434",
                        help="LLM base URL")
    parser.add_argument("--llm_model_name", type=str, default="qwen3:8B",
                        help="LLM model name")

    # 嵌入模型配置
    parser.add_argument("--embedding_model", type=str,
                        default="sentence-transformers/msmarco-distilbert-base-tas-b",
                        help="Sentence embedding model name")

    # 搜索配置
    parser.add_argument("--max_depth", type=int, default=5,
                        help="Maximum search depth")
    parser.add_argument("--relation_search_width", type=int, default=10,
                        help="Number of relations to retrieve per entity")
    parser.add_argument("--entity_search_width", type=int, default=5,
                        help="Number of entities to retrieve per relation")

    # 剪枝配置
    parser.add_argument("--remove_unnecessary_rel", action="store_true", default=True,
                        help="Whether to remove unnecessary relations")
    parser.add_argument("--temperature_exploration", type=float, default=0.3,
                        help="Temperature for exploration stage")
    parser.add_argument("--prune_tools", type=str, default="llm",
                        choices=["llm", "bm25", "sentencebert"],
                        help="Pruning tools")

    # 输入输出配置
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input JSON file containing questions")
    parser.add_argument("--output_file", type=str, default="results.jsonl",
                        help="Output file for results")
    parser.add_argument("--question", type=str, default="What entity is reached from e1 via r2 and then r4?",
                        help="Single question to process (if no input file)")
    parser.add_argument("--start_entity_id", type=str, default="e1",
                        help="Starting entity ID for single question mode")

    return parser.parse_args()


def initialize_components(args: argparse.Namespace) -> Tuple[Any, Any, SentenceTransformer]:
    """初始化数据库、LLM和嵌入模型"""
    # 初始化数据库连接
    dao = KGDao(
        args.neo4j_url,
        args.neo4j_username,
        args.neo4j_password,
        args.neo4j_database
    )

    # 测试数据库连接
    try:
        result = dao.execute("MATCH (n) RETURN n LIMIT 1")
        if not result:
            print("数据库连接失败，请检查数据库配置。")
            exit(1)
    except Exception as e:
        print(f"数据库连接失败: {e}")
        exit(1)

    # 初始化LLM
    chat_model = ChatModel(
        provider=args.llm_provider,
        model_name=args.llm_model_name,
        base_url=args.llm_base_url
    )

    # 初始化嵌入模型
    model = SentenceTransformer(args.embedding_model)

    return dao, chat_model, model


def prepare_input(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """准备输入数据"""
    if args.input_file:
        # 从文件读取
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"输入文件不存在: {args.input_file}")

        with open(args.input_file, 'r', encoding='utf-8') as f:
            if args.input_file.endswith('.jsonl'):
                data = [json.loads(line.strip()) for line in f if line.strip()]
            else:
                data = json.load(f)

        return data
    else:
        # 单问题模式
        if not args.question:
            raise ValueError("必须提供 --question 或 --input_file")

        data = [{
            'question': args.question,
            'topic_entity': args.start_entity_id,
            'topic_entity_name': args.start_entity_id
        }]
        return data


def process_single_question(
        question: str,
        topic_entity_id: str,
        topic_entity_name: str,
        dao: KGDao,
        chat_model: ChatModel,
        embedding_model: SentenceTransformer,
        args: argparse.Namespace
) -> Dict[str, Any]:
    """处理单个问题的推理过程"""

    # 初始化实体
    eids = [topic_entity_id]
    enames = [topic_entity_name]
    eid2name = dict(zip(eids, enames))
    name2eid = dict(zip(enames, eids))

    # 初始化状态变量
    pre_relations = []
    pre_heads = [-1] * len(eids)
    cluster_chain_of_entities = []

    print(f"开始 QA 循环，问题: {question}")
    print(f"初始实体: {eid2name}")

    # 推理循环
    for depth in range(1, args.max_depth + 1):
        print(f"\n=== 跳数深度 {depth} ===")

        # 步骤A: 关系搜索
        current_entity_relations_list = []
        for i, eid in enumerate(eids):
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

        # 步骤B: 字典构建
        ent_rel_ent_dict = {}
        for ent_rel in current_entity_relations_list:
            curr_entity_id = ent_rel['entity']
            relation = ent_rel['relation']
            is_head = ent_rel['head']
            direction = 'head' if is_head else 'tail'

            # 搜索相邻实体
            found_entities = utils.entity_search(
                dao, curr_entity_id, relation, is_head
            )

            # 更新映射
            found_ids = []
            for item in found_entities:
                f_id = item['id']
                f_name = item['name']
                eid2name[f_id] = f_name
                name2eid[f_name] = f_id
                found_ids.append(f_id)

            # 构建字典结构
            if curr_entity_id not in ent_rel_ent_dict:
                ent_rel_ent_dict[curr_entity_id] = {}
            if direction not in ent_rel_ent_dict[curr_entity_id]:
                ent_rel_ent_dict[curr_entity_id][direction] = {}
            ent_rel_ent_dict[curr_entity_id][direction][relation] = found_ids

        # 步骤C: 剪枝与推理
        flag, chain_of_entities, entities_id, filter_relations, filter_head, new_ent_rel_ent_dict = utils.entity_condition_prune(
            question,
            ent_rel_ent_dict,
            eid2name,
            name2eid,
            chat_model,
            embedding_model
        )

        # 步骤D: 状态更新
        cluster_chain_of_entities.append(chain_of_entities)

        eids = entities_id
        pre_relations = filter_relations
        pre_heads = filter_head

        print(f"深度 {depth} 结果 - 继续: {flag}, 实体数量: {len(eids)}")

        if not flag:
            print("推理循环因剪枝条件终止。")
            break

    # 构建结果
    result = {
        'question': question,
        'topic_entity': topic_entity_id,
        'chain_of_entities': cluster_chain_of_entities,
        'final_entities': eids,
        'depth': len(cluster_chain_of_entities)
    }

    return result


def save_results(results: List[Dict[str, Any]], output_file: str):
    """保存结果到文件（可选，不影响输出）"""
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    """主函数"""
    args = parse_args()

    try:
        # 初始化组件
        dao, chat_model, embedding_model = initialize_components(args)

        # 准备输入数据
        data = prepare_input(args)

        # 处理所有问题
        results = []

        for item in data:
            question = item['question']
            topic_entity_id = item.get('topic_entity', args.start_entity_id)
            topic_entity_name = item.get('topic_entity_name', topic_entity_id)

            try:
                result = process_single_question(
                    question, topic_entity_id, topic_entity_name,
                    dao, chat_model, embedding_model, args
                )
                results.append(result)
            except Exception as e:
                print(f"\n问题处理失败: {question}")
                print(f"错误: {e}")
                results.append({
                    'question': question,
                    'error': str(e),
                    'success': False
                })

        # 保存结果（静默保存，不影响输出）
        save_results(results, args.output_file)

        # 打印最终链条（与第一版格式一致）
        print("\n=== 最终链条 ===")
        for result in results:
            for chain in result['chain_of_entities']:
                print(chain)

    except Exception as e:
        print(f"\n程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
