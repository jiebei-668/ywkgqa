cypher = """// 1. 先创建/匹配所有节点 (e1 到 e6)
// 确保每个节点都有 id 和 name 且值相同
MERGE (e1:Entity {id: 'e1', name: 'e1'})
MERGE (e2:Entity {id: 'e2', name: 'e2'})
MERGE (e3:Entity {id: 'e3', name: 'e3'})
MERGE (e4:Entity {id: 'e4', name: 'e4'})
MERGE (e5:Entity {id: 'e5', name: 'e5'})
MERGE (e6:Entity {id: 'e6', name: 'e6'})

// 2. 创建第一条路径的关系: e6->r5->e1->r1->e2->r3->e4
// 关系 r5: e6 -> e1
MERGE (e6)-[:RELATION {id: 'r5', name: 'r5'}]->(e1)
// 关系 r1: e1 -> e2
MERGE (e1)-[:RELATION {id: 'r1', name: 'r1'}]->(e2)
// 关系 r3: e2 -> e4
MERGE (e2)-[:RELATION {id: 'r3', name: 'r3'}]->(e4)

// 3. 创建第二条路径的关系: e1->r2->e3->r4->e5
// 注意：e1 已经在上面被引用，这里直接复用
// 关系 r2: e1 -> e3
MERGE (e1)-[:RELATION {id: 'r2', name: 'r2'}]->(e3)
// 关系 r4: e3 -> e5
MERGE (e3)-[:RELATION {id: 'r4', name: 'r4'}]->(e5)"""
