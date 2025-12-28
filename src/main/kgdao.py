from neo4j import GraphDatabase


class KGDao:
    def __init__(self, uri, username, password, database):
        """
        初始化数据库连接
        :param uri: 数据库地址，如 "bolt://localhost:7687"
        :param username: 用户名
        :param password: 密码
        :param database: 数据库名称
        """
        self.database = database
        try:
            # 创建驱动
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # 验证连接是否成功
            self.driver.verify_connectivity()
            
        except Exception as e:
            raise e

    def execute(self, cypher):
        """
        执行 Cypher 查询并返回原始 JSON 格式数据
        :param cypher: Cypher 查询语句字符串
        :return: List[Dict]，即包含字典的列表
        """
        results = []
        try:
            # 使用上下文管理器自动处理 session 的关闭
            with self.driver.session(database=self.database) as session:
                result_obj = session.run(cypher)
                
                # 将 Neo4j 的 Record 对象转换为 Python 原生字典 (JSON 格式)
                # record.data() 是 neo4j 驱动自带的方法，能自动将节点和关系转为字典
                results = [record.data() for record in result_obj]
                
        except Exception as e:
            # 这里可以选择抛出异常，或者返回空列表，视业务需求而定
            # raise e 
            return []

        return results

    def close(self):
        """
        手动关闭驱动（一般在程序退出时调用）
        """
        if self.driver:
            self.driver.close()

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 配置你的连接信息
    URI = "neo4j://localhost:7687"
    USER = "neo4j"
    PASSWORD = "KOBEforever668!"
    DB_NAME = "libai"

    try:
        # 1. 实例化 DAO
        kg_dao = KGDao(URI, USER, PASSWORD, DB_NAME)

        # 2. 编写查询语句 (示例：查询前5个节点)
        cypher_query = "MATCH (n) RETURN n LIMIT 5"

        # 3. 执行查询
        data = kg_dao.execute(cypher_query)

        # 4. 打印结果 (JSON 格式)
        import json
        # print(json.dumps(data, indent=4, ensure_ascii=False))
        print(json.dumps(data, indent=4, ensure_ascii=False, default=str))

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 5. 关闭连接
        if 'kg_dao' in locals():
            kg_dao.close()
