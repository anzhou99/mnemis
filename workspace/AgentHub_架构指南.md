# AgentHub 架构指南

> 版本 v1.0 | 适用人群：Java 学习者 | 项目规模：单体 Spring Boot 应用

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术选型](#2-技术选型)
3. [整体分层架构](#3-整体分层架构)
4. [模块划分与职责](#4-模块划分与职责)
5. [包结构设计](#5-包结构设计)
6. [各层详细规范](#6-各层详细规范)
7. [AI 框架适配层设计（核心解耦点）](#7-ai-框架适配层设计核心解耦点)
8. [数据库表设计](#8-数据库表设计)
9. [核心流程时序说明](#9-核心流程时序说明)
10. [模块间依赖规则](#10-模块间依赖规则)
11. [配置管理规范](#11-配置管理规范)
12. [开发阶段建议](#12-开发阶段建议)

---

## 1. 项目概述

AgentHub 是一个 AI Agent 管理与运行平台。用户可以创建知识库、配置工具、发起对话，并让 Agent 自主规划和执行多步骤任务。

### 1.1 核心功能模块

| 模块 | 说明 |
|------|------|
| 用户认证模块 | 注册、登录、JWT 鉴权 |
| 用户管理模块 | 用户信息 CRUD、角色管理（普通接口练习） |
| 对话引擎模块 | 多轮对话、SSE 流式输出、历史管理 |
| 工具调用模块 | Function Calling、工具注册与扩展 |
| 知识库模块 | 文档上传、RAG 流水线、向量检索 |
| Agent 编排模块 | ReAct 规划执行、多步推理 |
| 任务管理模块 | 异步任务提交、状态追踪、日志回放 |

### 1.2 设计目标

- **分层清晰**：Controller → Service → Repository 三层严格分离，各层职责单一
- **模块解耦**：业务模块之间通过接口通信，禁止跨模块直接调用 Repository
- **AI 框架可替换**：通过适配层隔离 Spring AI 与 LangChain4j，切换时只改实现不改业务
- **易于扩展**：工具、LLM 接入方均基于策略模式，新增时无需修改已有代码

---

## 2. 技术选型

### 2.1 基础框架

| 技术 | 版本建议 | 用途 |
|------|----------|------|
| Spring Boot | 3.2.x | 应用框架 |
| Spring Security | 6.x | 认证与授权 |
| MyBatis-Plus | 3.5.x | ORM，简化 CRUD |
| MySQL | 8.0+ | 主数据库 |
| Redis | 7.x | 缓存、限流、会话 |
| pgvector / Milvus | — | 向量数据库（二选一） |

> **pgvector 推荐新手**：作为 PostgreSQL 插件，无需额外部署，适合学习阶段。Milvus 性能更强，适合生产场景。

### 2.2 AI 框架（两套，按需切换）

| 框架 | 版本 | 特点 |
|------|------|------|
| Spring AI | 1.0.x | Spring 生态原生，注解驱动，上手快 |
| LangChain4j | 0.31.x | 功能更完整，Agent/RAG 抽象更成熟 |

### 2.3 辅助工具

| 工具 | 用途 |
|------|------|
| Apache PDFBox / Tika | 文档解析 |
| MapStruct | DTO ↔ Entity 对象转换 |
| Lombok | 减少样板代码 |
| Knife4j | 接口文档（基于 Swagger） |
| Docker Compose | 本地环境一键启动 |

---

## 3. 整体分层架构

```
┌─────────────────────────────────────────────────────────┐
│                      客户端层                            │
│              浏览器 / Postman / 前端应用                  │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP / SSE / WebSocket
┌──────────────────────────▼──────────────────────────────┐
│                    接入层 Interface Layer                 │
│   Controller（REST 接口）/ GlobalExceptionHandler        │
│   请求校验（@Valid）/ 统一响应封装（Result<T>）           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   业务层 Service Layer                    │
│                                                         │
│  ┌───────────┐  ┌──────────┐  ┌───────────┐            │
│  │ AuthService│  │UserService│  │ChatService│  ...       │
│  └───────────┘  └──────────┘  └───────────┘            │
│                                                         │
│  业务规则 / 事务控制 / 跨模块协调（通过 Service 接口）    │
└────────────┬───────────────────────────┬────────────────┘
             │                           │
┌────────────▼────────────┐ ┌────────────▼────────────────┐
│    数据层 Repository     │ │    AI 适配层 AI Adapter      │
│                         │ │                             │
│  Mapper（MyBatis-Plus）  │ │  AiChatPort（接口）          │
│  Entity / PO            │ │  ├─ SpringAiAdapter（实现）  │
│  MySQL / Redis          │ │  └─ LangChain4jAdapter（实现）│
└─────────────────────────┘ │                             │
                            │  EmbeddingPort（接口）       │
                            │  VectorStorePort（接口）     │
                            └─────────────────────────────┘
```

### 3.1 分层职责说明

**接入层（Interface Layer）**
- 只做：参数接收、参数校验、调用 Service、封装响应
- 不做：任何业务逻辑、不直接访问数据库、不直接调用 AI 框架

**业务层（Service Layer）**
- 只做：业务规则、事务管理、跨模块协调、调用 Repository 和 AI 适配层
- 不做：构造 HTTP 响应、拼接 SQL、直接操作 AI SDK 原生对象

**数据层（Repository Layer）**
- 只做：数据库 CRUD、缓存读写
- 不做：业务判断、调用其他 Service

**AI 适配层（AI Adapter Layer）**
- 只做：封装 AI SDK 调用细节，向上暴露统一接口
- 不做：业务规则，不了解"这是订单"还是"这是聊天"

---

## 4. 模块划分与职责

### 4.1 用户认证模块（auth）

**职责**：处理身份认证，签发与验证 JWT。

```
对外接口：
  POST /api/auth/register    注册
  POST /api/auth/login       登录，返回 access_token + refresh_token
  POST /api/auth/refresh     刷新 Token
  POST /api/auth/logout      登出（Token 加入 Redis 黑名单）

核心类：
  AuthController             接收请求
  AuthService                注册/登录业务逻辑
  JwtUtils                   Token 生成与解析工具类
  JwtAuthenticationFilter    Spring Security 过滤器，拦截所有请求做 Token 校验
  UserDetailsServiceImpl     实现 Spring Security 的用户加载接口
```

**关键设计点**：

- `JwtAuthenticationFilter` 继承 `OncePerRequestFilter`，每个请求只执行一次
- Token 黑名单存于 Redis，Key 为 `blacklist:token:{jti}`，TTL 与 Token 过期时间对齐
- Refresh Token 与 Access Token 分离：Access Token 有效期 2 小时，Refresh Token 有效期 7 天

---

### 4.2 用户管理模块（user）

**职责**：用户信息的增删改查，角色管理。这是练习标准 CRUD 接口的主要场所。

```
对外接口：
  GET    /api/users              分页查询用户列表（管理员）
  GET    /api/users/{id}         查询单个用户
  PUT    /api/users/{id}         修改用户信息
  DELETE /api/users/{id}         删除用户（逻辑删除）
  PUT    /api/users/{id}/role    修改用户角色（管理员）
  GET    /api/users/me           查询当前登录用户信息
  PUT    /api/users/me/password  修改密码

核心类：
  UserController             接收请求，参数校验
  UserService / UserServiceImpl  业务逻辑
  UserMapper                 MyBatis-Plus Mapper
  UserEntity                 数据库实体（PO）
  UserVO                     返回给前端的视图对象
  UserUpdateDTO              接收前端修改请求的数据传输对象
```

**关键设计点**：

- 分页查询使用 MyBatis-Plus 的 `Page<T>` + `IPage<T>`，不要手写分页 SQL
- 逻辑删除：Entity 上加 `@TableLogic` 注解，`deleted` 字段为 0/1
- `UserVO` 中不包含密码字段，通过 MapStruct 映射时自动排除
- 当前用户 ID 从 `SecurityContextHolder` 中获取，封装成 `CurrentUserUtils` 工具类

---

### 4.3 对话引擎模块（chat）

**职责**：管理对话会话，调用 LLM 实现多轮对话，支持 SSE 流式输出。

```
对外接口：
  POST /api/conversations              创建新会话
  GET  /api/conversations              查询当前用户的会话列表
  GET  /api/conversations/{id}/messages  获取历史消息
  POST /api/conversations/{id}/messages  发送消息（普通响应）
  GET  /api/conversations/{id}/stream   发送消息（SSE 流式响应）
  DELETE /api/conversations/{id}        删除会话

核心类：
  ChatController             接收请求，处理 SSE 连接
  ConversationService        会话创建与管理
  ChatService                核心：组装历史、调用 AI 适配层、持久化
  MessageRepository          消息存储
  ContextWindowManager       历史消息裁剪（Token 限制管理）
```

**关键设计点**：

```
SSE 流式实现思路（Spring MVC）：

Controller 返回 SseEmitter：
  SseEmitter emitter = new SseEmitter(60_000L);  // 60秒超时
  chatService.streamChat(request, emitter);       // 异步执行
  return emitter;

ChatService 在独立线程中：
  1. 调用 AiChatPort.streamChat()，得到 Flux<String> 或 Stream<String>
  2. 每收到一个 token：emitter.send(token)
  3. 完成后：emitter.complete()
  4. 异常时：emitter.completeWithError(e)
```

- `ContextWindowManager` 负责控制历史消息数量，超出时按"保留最新 N 轮"策略裁剪
- 每条消息存储时记录 `role`（user / assistant / system）和 `content`

---

### 4.4 工具调用模块（tool）

**职责**：定义和管理 Agent 可调用的工具，负责工具的注册、路由与执行。

```
对外接口（管理接口）：
  GET /api/tools           查询所有可用工具列表及其描述

内部接口（供 Agent 编排模块调用）：
  ToolRegistry.getTool(name) → ToolHandler
  ToolExecutor.execute(name, params) → ToolResult

内置工具：
  WebSearchTool            调用搜索 API（如 Serper / Tavily）
  WeatherTool              查询天气
  CalculatorTool           数学计算
  DatabaseQueryTool        查询应用内数据库（受限 SQL）
  CurrentTimeTool          获取当前时间
```

**关键设计点**：

```java
// 工具统一接口（策略模式）
public interface ToolHandler {
    String getName();           // 工具名，全局唯一
    String getDescription();    // 工具描述，传给 LLM 用于决策
    JsonSchema getInputSchema(); // 入参 JSON Schema
    ToolResult execute(Map<String, Object> params);
}

// 工具注册中心（Spring 启动时自动扫描所有 ToolHandler Bean）
@Component
public class ToolRegistry {
    private final Map<String, ToolHandler> tools;

    public ToolRegistry(List<ToolHandler> handlers) {
        this.tools = handlers.stream()
            .collect(Collectors.toMap(ToolHandler::getName, h -> h));
    }

    public ToolHandler getTool(String name) {
        return Optional.ofNullable(tools.get(name))
            .orElseThrow(() -> new ToolNotFoundException(name));
    }
}
```

- 新增工具只需实现 `ToolHandler` 接口并加 `@Component`，无需修改任何已有代码（开闭原则）
- `ToolResult` 包含 `success`、`content`、`errorMessage` 字段
- 工具执行要有超时控制（`@Async` + `Future.get(timeout)`）

---

### 4.5 知识库模块（knowledge）

**职责**：文档上传、解析、向量化，提供语义检索接口（RAG 核心）。

```
对外接口：
  POST   /api/knowledge-bases              创建知识库
  GET    /api/knowledge-bases              查询用户的知识库列表
  DELETE /api/knowledge-bases/{id}         删除知识库
  POST   /api/knowledge-bases/{id}/documents  上传文档
  GET    /api/knowledge-bases/{id}/documents  查询文档列表
  DELETE /api/knowledge-bases/{id}/documents/{docId}  删除文档

内部接口（供 ChatService / Agent 调用）：
  KnowledgeRetrievalService.search(kbId, query, topK) → List<RelevantChunk>

核心类：
  DocumentProcessor          文档解析与分块流水线（模板方法模式）
  EmbeddingService           调用 EmbeddingPort 生成向量
  VectorStoreService         向量存储与检索（调用 VectorStorePort）
  ChunkingStrategy           分块策略接口（固定大小 / 段落 / 递归）
```

**关键设计点**：

```
RAG 写入流水线（文档上传时触发）：
  原始文件
    ↓ DocumentParser（PDFBox / Tika）
  纯文本
    ↓ ChunkingStrategy（按段落，overlap=100字符）
  List<TextChunk>
    ↓ EmbeddingPort.embed(texts) → List<float[]>
  向量列表
    ↓ VectorStorePort.save(chunks + vectors)
  存入向量数据库

RAG 检索流水线（对话时触发）：
  用户问题
    ↓ EmbeddingPort.embed(question) → float[]
  问题向量
    ↓ VectorStorePort.search(vector, topK=5)
  相关文本块
    ↓ 注入 Prompt（"根据以下内容回答：{context}\n\n问题：{question}"）
  传给 LLM
```

- `DocumentProcessor` 用模板方法模式定义流水线，子类只需覆盖解析步骤
- 分块大小建议：512 token，重叠 50 token，具体参数通过配置文件调整
- 向量数据库中每条记录除向量外，还存储 `knowledge_base_id`、`document_id`、`chunk_index`、原始文本

---

### 4.6 Agent 编排模块（agent）

**职责**：实现 ReAct 模式的 Agent 执行引擎，协调 LLM 推理与工具调用。

```
对外接口：
  POST /api/agents              创建 Agent 配置（绑定工具和知识库）
  GET  /api/agents              查询 Agent 列表
  POST /api/agents/{id}/run     运行 Agent（提交任务）
  GET  /api/agents/{id}/runs/{runId}  查询运行状态

核心类：
  AgentOrchestrator          ReAct 主循环执行器
  AgentContext               单次运行的上下文（历史步骤、工具调用记录）
  ReActStepParser            解析 LLM 输出的 Thought/Action/Action Input
  AgentRunRecorder           记录每个执行步骤到数据库
```

**关键设计点**：

```
ReAct 执行循环：

while (步骤数 < 最大步骤 && !finished) {
    1. 构造 Prompt：
       System Prompt（角色定义 + 工具列表描述）
       + 历史步骤（Thought + Action + Observation）
       + 当前目标

    2. 调用 AiChatPort.chat() → LLM 输出

    3. 解析输出：
       - 包含 "Final Answer:" → 提取最终答案，finished = true
       - 包含 "Action:" → 提取工具名和参数

    4. 若是工具调用：
       ToolResult result = ToolExecutor.execute(action, params)
       将 Observation 追加到历史

    5. 记录步骤到数据库（AgentRunRecorder）
}

若超出最大步骤数 → 返回超时错误
```

- `ReActStepParser` 要健壮处理 LLM 输出格式不稳定的情况（正则 + fallback）
- 每个执行步骤持久化，支持前端轮询查看执行轨迹
- Agent 配置中记录：允许使用的工具列表、关联的知识库 ID、系统 Prompt 模板

---

### 4.7 任务管理模块（task）

**职责**：异步任务的生命周期管理，Agent 长时任务的提交、查询与取消。

```
对外接口：
  GET    /api/tasks          查询当前用户的任务列表
  GET    /api/tasks/{id}     查询任务详情（含执行步骤）
  DELETE /api/tasks/{id}     取消正在执行的任务
  GET    /api/tasks/{id}/sse  订阅任务进度（SSE 推送）

核心类：
  TaskService                任务提交与状态管理
  TaskExecutor               线程池封装，提交异步任务
  TaskStatusPusher           通过 SseEmitter 推送状态变更
```

**关键设计点**：

```java
// 自定义线程池（不使用默认线程池，便于监控和控制）
@Bean("agentTaskExecutor")
public ThreadPoolExecutor agentTaskExecutor() {
    return new ThreadPoolExecutor(
        4,                                    // 核心线程数
        10,                                   // 最大线程数
        60, TimeUnit.SECONDS,                 // 空闲线程存活时间
        new LinkedBlockingQueue<>(100),       // 队列容量
        new ThreadPoolExecutor.CallerRunsPolicy() // 拒绝策略：调用者线程执行
    );
}
```

- 任务状态枚举：`PENDING → RUNNING → SUCCEEDED / FAILED / CANCELLED`
- 取消任务：在 `task` 表标记 `cancel_requested = true`，执行线程在每步结束后检查该标志
- 前端通过 SSE 订阅任务进度，服务端在状态变更时主动推送

---

## 5. 包结构设计

```
com.agenthub
├── common                         # 公共组件（无业务含义）
│   ├── result                     # 统一响应封装
│   │   ├── Result.java
│   │   └── ResultCode.java
│   ├── exception                  # 自定义异常体系
│   │   ├── BusinessException.java
│   │   ├── NotFoundException.java
│   │   └── GlobalExceptionHandler.java
│   ├── utils                      # 工具类
│   │   ├── JwtUtils.java
│   │   ├── CurrentUserUtils.java
│   │   └── PageUtils.java
│   └── config                     # 全局配置
│       ├── SecurityConfig.java
│       ├── RedisConfig.java
│       └── ThreadPoolConfig.java
│
├── ai                             # AI 适配层（核心解耦）
│   ├── port                       # 端口接口（抽象）
│   │   ├── AiChatPort.java
│   │   ├── EmbeddingPort.java
│   │   └── VectorStorePort.java
│   ├── springai                   # Spring AI 实现
│   │   ├── SpringAiChatAdapter.java
│   │   ├── SpringAiEmbeddingAdapter.java
│   │   └── SpringAiVectorStoreAdapter.java
│   ├── langchain4j                # LangChain4j 实现
│   │   ├── Lc4jChatAdapter.java
│   │   ├── Lc4jEmbeddingAdapter.java
│   │   └── Lc4jVectorStoreAdapter.java
│   └── model                      # AI 层专用模型
│       ├── ChatMessage.java
│       ├── ChatResponse.java
│       └── EmbeddingResult.java
│
├── module                         # 业务模块
│   ├── auth                       # 认证模块
│   │   ├── controller
│   │   │   └── AuthController.java
│   │   ├── service
│   │   │   ├── AuthService.java
│   │   │   └── impl
│   │   │       └── AuthServiceImpl.java
│   │   ├── dto
│   │   │   ├── LoginRequest.java
│   │   │   └── TokenResponse.java
│   │   └── security
│   │       ├── JwtAuthenticationFilter.java
│   │       └── UserDetailsServiceImpl.java
│   │
│   ├── user                       # 用户管理模块
│   │   ├── controller
│   │   │   └── UserController.java
│   │   ├── service
│   │   │   ├── UserService.java
│   │   │   └── impl
│   │   │       └── UserServiceImpl.java
│   │   ├── mapper
│   │   │   └── UserMapper.java
│   │   ├── entity
│   │   │   └── UserEntity.java
│   │   ├── dto
│   │   │   ├── UserUpdateDTO.java
│   │   │   └── UserPageQueryDTO.java
│   │   ├── vo
│   │   │   └── UserVO.java
│   │   └── converter
│   │       └── UserConverter.java  # MapStruct 接口
│   │
│   ├── chat                       # 对话引擎模块
│   │   ├── controller
│   │   │   └── ChatController.java
│   │   ├── service
│   │   │   ├── ConversationService.java
│   │   │   ├── ChatService.java
│   │   │   ├── ContextWindowManager.java
│   │   │   └── impl
│   │   ├── mapper
│   │   │   ├── ConversationMapper.java
│   │   │   └── MessageMapper.java
│   │   ├── entity
│   │   │   ├── ConversationEntity.java
│   │   │   └── MessageEntity.java
│   │   ├── dto
│   │   └── vo
│   │
│   ├── tool                       # 工具调用模块
│   │   ├── registry
│   │   │   ├── ToolHandler.java    # 接口
│   │   │   ├── ToolRegistry.java
│   │   │   └── ToolExecutor.java
│   │   ├── handler                # 各工具实现
│   │   │   ├── WebSearchTool.java
│   │   │   ├── WeatherTool.java
│   │   │   ├── CalculatorTool.java
│   │   │   └── CurrentTimeTool.java
│   │   ├── model
│   │   │   ├── ToolResult.java
│   │   │   └── JsonSchema.java
│   │   └── controller
│   │       └── ToolController.java
│   │
│   ├── knowledge                  # 知识库模块
│   │   ├── controller
│   │   │   └── KnowledgeController.java
│   │   ├── service
│   │   │   ├── KnowledgeBaseService.java
│   │   │   ├── DocumentProcessService.java
│   │   │   ├── KnowledgeRetrievalService.java
│   │   │   └── impl
│   │   ├── pipeline
│   │   │   ├── DocumentProcessor.java  # 模板方法抽象类
│   │   │   ├── PdfDocumentProcessor.java
│   │   │   ├── TextDocumentProcessor.java
│   │   │   └── chunking
│   │   │       ├── ChunkingStrategy.java  # 接口
│   │   │       ├── FixedSizeChunking.java
│   │   │       └── ParagraphChunking.java
│   │   ├── mapper
│   │   │   ├── KnowledgeBaseMapper.java
│   │   │   └── DocumentMapper.java
│   │   └── entity
│   │
│   ├── agent                      # Agent 编排模块
│   │   ├── controller
│   │   │   └── AgentController.java
│   │   ├── service
│   │   │   ├── AgentService.java
│   │   │   └── impl
│   │   ├── orchestrator
│   │   │   ├── AgentOrchestrator.java
│   │   │   ├── AgentContext.java
│   │   │   └── ReActStepParser.java
│   │   ├── mapper
│   │   └── entity
│   │
│   └── task                       # 任务管理模块
│       ├── controller
│       │   └── TaskController.java
│       ├── service
│       │   ├── TaskService.java
│       │   └── impl
│       ├── executor
│       │   └── TaskExecutor.java
│       ├── pusher
│       │   └── TaskStatusPusher.java
│       ├── mapper
│       └── entity
│           └── TaskEntity.java
│
└── AgentHubApplication.java
```

---

## 6. 各层详细规范

### 6.1 Controller 层规范

```java
@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;

    // 1. 方法只做：接收参数 → 调用 Service → 返回 Result
    // 2. 参数校验用 @Valid，不手写 if 判断
    // 3. 所有方法返回 Result<T>，不直接返回业务对象
    // 4. 不写任何业务逻辑

    @GetMapping
    public Result<IPage<UserVO>> pageUsers(@Valid UserPageQueryDTO query) {
        return Result.ok(userService.pageUsers(query));
    }

    @GetMapping("/{id}")
    public Result<UserVO> getUser(@PathVariable Long id) {
        return Result.ok(userService.getUserById(id));
    }

    @PutMapping("/{id}")
    public Result<Void> updateUser(@PathVariable Long id,
                                   @Valid @RequestBody UserUpdateDTO dto) {
        userService.updateUser(id, dto);
        return Result.ok();
    }

    @DeleteMapping("/{id}")
    public Result<Void> deleteUser(@PathVariable Long id) {
        userService.deleteUser(id);
        return Result.ok();
    }
}
```

### 6.2 Service 层规范

```java
// 接口与实现分离，接口放 service 包，实现放 service/impl 包
public interface UserService {
    IPage<UserVO> pageUsers(UserPageQueryDTO query);
    UserVO getUserById(Long id);
    void updateUser(Long id, UserUpdateDTO dto);
    void deleteUser(Long id);
}

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)  // 默认只读事务，写操作单独加 @Transactional
public class UserServiceImpl implements UserService {

    private final UserMapper userMapper;
    private final UserConverter userConverter;  // MapStruct

    @Override
    public UserVO getUserById(Long id) {
        UserEntity user = userMapper.selectById(id);
        if (user == null) {
            throw new NotFoundException("用户不存在：" + id);
        }
        return userConverter.toVO(user);  // Entity → VO 转换在 Service 层完成
    }

    @Override
    @Transactional  // 写操作显式开启事务
    public void deleteUser(Long id) {
        // 业务规则在 Service 中
        UserEntity user = userMapper.selectById(id);
        if (user == null) throw new NotFoundException("用户不存在");
        if (user.isAdmin()) throw new BusinessException("不能删除管理员账号");
        userMapper.deleteById(id);  // 逻辑删除，MyBatis-Plus 自动处理
    }
}
```

### 6.3 统一响应结构

```java
@Data
public class Result<T> {
    private int code;
    private String message;
    private T data;

    public static <T> Result<T> ok(T data) {
        return new Result<>(200, "success", data);
    }

    public static <T> Result<T> ok() {
        return ok(null);
    }

    public static <T> Result<T> fail(int code, String message) {
        return new Result<>(code, message, null);
    }
}

// 响应示例：
// { "code": 200, "message": "success", "data": { "id": 1, "username": "tom" } }
// { "code": 404, "message": "用户不存在：1", "data": null }
```

### 6.4 全局异常处理

```java
@RestControllerAdvice
@Slf4j
public class GlobalExceptionHandler {

    // 业务异常（主动抛出的，不打 ERROR 日志）
    @ExceptionHandler(BusinessException.class)
    public Result<Void> handleBusiness(BusinessException e) {
        return Result.fail(e.getCode(), e.getMessage());
    }

    // 参数校验失败
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public Result<Void> handleValidation(MethodArgumentNotValidException e) {
        String msg = e.getBindingResult().getFieldErrors().stream()
            .map(FieldError::getDefaultMessage)
            .collect(Collectors.joining(", "));
        return Result.fail(400, msg);
    }

    // 未知异常（打 ERROR 日志，返回通用错误）
    @ExceptionHandler(Exception.class)
    public Result<Void> handleUnknown(Exception e) {
        log.error("Unexpected error", e);
        return Result.fail(500, "服务内部错误，请稍后重试");
    }
}
```

---

## 7. AI 框架适配层设计（核心解耦点）

这是整个架构中最重要的设计决策。通过端口-适配器模式（六边形架构思想），将 AI 框架的具体实现与业务逻辑彻底分离。

### 7.1 端口接口定义

```java
// 端口1：对话能力
public interface AiChatPort {

    /**
     * 普通对话（阻塞返回完整响应）
     */
    ChatResponse chat(List<ChatMessage> messages);

    /**
     * 流式对话（返回响应流，由调用方消费）
     * 返回 Flux<String> 需要 Spring WebFlux 依赖，
     * 不想引入响应式可改为接受 Consumer<String> 回调
     */
    void streamChat(List<ChatMessage> messages, Consumer<String> onToken,
                    Runnable onComplete, Consumer<Throwable> onError);

    /**
     * 带工具描述的对话（Function Calling）
     */
    ChatResponse chatWithTools(List<ChatMessage> messages, List<ToolDefinition> tools);
}

// 端口2：向量化能力
public interface EmbeddingPort {
    float[] embed(String text);
    List<float[]> embedBatch(List<String> texts);
}

// 端口3：向量存储能力
public interface VectorStorePort {
    void save(String collection, List<VectorDocument> documents);
    List<VectorDocument> search(String collection, float[] queryVector, int topK);
    void delete(String collection, List<String> documentIds);
}

// 共享模型类（不依赖任何 AI 框架的 POJO）
@Data @AllArgsConstructor
public class ChatMessage {
    private String role;    // "system" / "user" / "assistant"
    private String content;
}
```

### 7.2 Spring AI 实现

```java
@Component
@ConditionalOnProperty(name = "agenthub.ai.provider", havingValue = "spring-ai")
@RequiredArgsConstructor
public class SpringAiChatAdapter implements AiChatPort {

    private final ChatClient chatClient;  // Spring AI 的 ChatClient

    @Override
    public ChatResponse chat(List<ChatMessage> messages) {
        // 将自定义 ChatMessage 转换为 Spring AI 的消息格式
        List<Message> springMessages = messages.stream()
            .map(this::toSpringMessage)
            .toList();

        Prompt prompt = new Prompt(springMessages);
        ChatResponse response = chatClient.call(prompt);

        return new ChatResponse(
            response.getResult().getOutput().getContent()
        );
    }

    @Override
    public void streamChat(List<ChatMessage> messages,
                           Consumer<String> onToken,
                           Runnable onComplete,
                           Consumer<Throwable> onError) {
        List<Message> springMessages = messages.stream()
            .map(this::toSpringMessage)
            .toList();

        chatClient.stream(new Prompt(springMessages))
            .doOnNext(chunk -> onToken.accept(
                chunk.getResult().getOutput().getContent()))
            .doOnComplete(onComplete)
            .doOnError(onError)
            .subscribe();
    }

    private Message toSpringMessage(ChatMessage msg) {
        return switch (msg.getRole()) {
            case "user" -> new UserMessage(msg.getContent());
            case "assistant" -> new AssistantMessage(msg.getContent());
            case "system" -> new SystemMessage(msg.getContent());
            default -> throw new IllegalArgumentException("Unknown role: " + msg.getRole());
        };
    }
}
```

### 7.3 LangChain4j 实现

```java
@Component
@ConditionalOnProperty(name = "agenthub.ai.provider", havingValue = "langchain4j")
@RequiredArgsConstructor
public class Lc4jChatAdapter implements AiChatPort {

    private final ChatLanguageModel chatModel;       // LangChain4j 模型
    private final StreamingChatLanguageModel streamingModel;

    @Override
    public ChatResponse chat(List<ChatMessage> messages) {
        List<dev.langchain4j.data.message.ChatMessage> lc4jMessages =
            messages.stream().map(this::toLc4jMessage).toList();

        Response<AiMessage> response = chatModel.generate(lc4jMessages);

        return new ChatResponse(response.content().text());
    }

    @Override
    public void streamChat(List<ChatMessage> messages,
                           Consumer<String> onToken,
                           Runnable onComplete,
                           Consumer<Throwable> onError) {
        List<dev.langchain4j.data.message.ChatMessage> lc4jMessages =
            messages.stream().map(this::toLc4jMessage).toList();

        streamingModel.generate(lc4jMessages, new StreamingResponseHandler<>() {
            @Override
            public void onNext(String token) { onToken.accept(token); }

            @Override
            public void onComplete(Response<AiMessage> response) { onComplete.run(); }

            @Override
            public void onError(Throwable error) { onError.accept(error); }
        });
    }

    private dev.langchain4j.data.message.ChatMessage toLc4jMessage(ChatMessage msg) {
        return switch (msg.getRole()) {
            case "user" -> new UserMessage(msg.getContent());
            case "assistant" -> new AiMessage(msg.getContent());
            case "system" -> new SystemMessage(msg.getContent());
            default -> throw new IllegalArgumentException("Unknown role: " + msg.getRole());
        };
    }
}
```

### 7.4 切换方式

只需修改 `application.yml` 中的一个配置项，无需改动任何业务代码：

```yaml
agenthub:
  ai:
    provider: spring-ai      # 切换为 langchain4j 只改这一行
    model: gpt-4o
    api-key: ${OPENAI_API_KEY}
```

---

## 8. 数据库表设计

### 8.1 用户相关

```sql
-- 用户表
CREATE TABLE t_user (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    username    VARCHAR(64) NOT NULL UNIQUE,
    password    VARCHAR(128) NOT NULL,           -- BCrypt 加密
    email       VARCHAR(128) UNIQUE,
    avatar_url  VARCHAR(256),
    role        VARCHAR(32) DEFAULT 'USER',      -- USER / ADMIN
    status      TINYINT DEFAULT 1,               -- 1正常 0禁用
    deleted     TINYINT DEFAULT 0,               -- 逻辑删除
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Token 黑名单（登出时写入 Redis，不需要数据库表）
-- Redis Key：blacklist:token:{jti}，Value：1，TTL：与 Token 过期时间一致
```

### 8.2 对话相关

```sql
-- 会话表
CREATE TABLE t_conversation (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id     BIGINT NOT NULL,
    title       VARCHAR(128),                    -- 会话标题（首条消息截取）
    agent_id    BIGINT,                          -- 若关联 Agent 则填此字段
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id)
);

-- 消息表
CREATE TABLE t_message (
    id              BIGINT PRIMARY KEY AUTO_INCREMENT,
    conversation_id BIGINT NOT NULL,
    role            VARCHAR(16) NOT NULL,        -- user / assistant / system
    content         TEXT NOT NULL,
    token_count     INT,                         -- 该消息消耗的 token 数（可选）
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_conversation_id (conversation_id)
);
```

### 8.3 知识库相关

```sql
-- 知识库表
CREATE TABLE t_knowledge_base (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id     BIGINT NOT NULL,
    name        VARCHAR(128) NOT NULL,
    description VARCHAR(512),
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id)
);

-- 文档表
CREATE TABLE t_document (
    id                BIGINT PRIMARY KEY AUTO_INCREMENT,
    knowledge_base_id BIGINT NOT NULL,
    filename          VARCHAR(256) NOT NULL,
    file_type         VARCHAR(32),              -- pdf / txt / md
    file_size         BIGINT,
    status            VARCHAR(32) DEFAULT 'PROCESSING',  -- PROCESSING / READY / FAILED
    chunk_count       INT DEFAULT 0,
    created_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_knowledge_base_id (knowledge_base_id)
);
```

### 8.4 Agent 与任务相关

```sql
-- Agent 配置表
CREATE TABLE t_agent (
    id              BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id         BIGINT NOT NULL,
    name            VARCHAR(128) NOT NULL,
    system_prompt   TEXT,
    tool_names      JSON,                        -- ["search","weather"]
    knowledge_base_ids JSON,                     -- [1,2,3]
    max_steps       INT DEFAULT 10,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 任务表
CREATE TABLE t_task (
    id               BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id          BIGINT NOT NULL,
    agent_id         BIGINT NOT NULL,
    goal             TEXT NOT NULL,              -- 用户输入的目标
    status           VARCHAR(32) DEFAULT 'PENDING',  -- PENDING/RUNNING/SUCCEEDED/FAILED/CANCELLED
    result           TEXT,                       -- 最终答案
    error_message    VARCHAR(512),
    cancel_requested TINYINT DEFAULT 0,          -- 取消标志位
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at       DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id)
);

-- 任务执行步骤表（ReAct 每步的记录）
CREATE TABLE t_task_step (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    task_id     BIGINT NOT NULL,
    step_index  INT NOT NULL,
    thought     TEXT,                            -- LLM 的推理过程
    action      VARCHAR(128),                    -- 调用的工具名
    action_input JSON,                           -- 工具入参
    observation TEXT,                            -- 工具返回结果
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_task_id (task_id)
);
```

---

## 9. 核心流程时序说明

### 9.1 SSE 流式对话流程

```
客户端                    ChatController              ChatService           AiChatPort（适配层）
  |                           |                           |                       |
  |-- GET /stream ----------->|                           |                       |
  |                           |-- 创建 SseEmitter ------->|                       |
  |                           |-- chatService.stream() -->|                       |
  |<-- 建立 SSE 连接 ----------|                           |                       |
  |                           |                           |-- 组装历史消息 -------->|
  |                           |                           |-- streamChat() ------>|
  |                           |                           |                       |-- 调用 LLM API
  |                           |                           |<-- onToken("你") ------|
  |<-- data: 你 ---------------|                           |                       |
  |                           |                           |<-- onToken("好") ------|
  |<-- data: 好 ---------------|                           |                       |
  |                           |                           |<-- onComplete() -------|
  |<-- data: [DONE] ----------|                           |                       |
  |                           |-- emitter.complete() ---->|                       |
```

### 9.2 RAG 文档写入流程

```
客户端 → KnowledgeController → DocumentProcessService
                                    ↓
                              DocumentProcessor（解析原始文件）
                                    ↓
                              ChunkingStrategy（分块）
                                    ↓
                              EmbeddingPort.embedBatch()（向量化）
                                    ↓
                              VectorStorePort.save()（存入向量库）
                                    ↓
                              DocumentMapper.updateStatus(READY)（更新状态）
```

### 9.3 Agent ReAct 执行流程

```
TaskController → TaskService → TaskExecutor（提交到线程池）
                                    ↓（异步线程）
                              AgentOrchestrator.run()
                                    |
                                    |-- 循环开始（max_steps 次）
                                    |     ↓
                                    |   构造 Prompt（目标 + 工具描述 + 历史步骤）
                                    |     ↓
                                    |   AiChatPort.chat()
                                    |     ↓
                                    |   ReActStepParser.parse(llmOutput)
                                    |     ↓
                                    |   if Final Answer → 结束
                                    |   if Action → ToolExecutor.execute()
                                    |     ↓
                                    |   AgentRunRecorder.recordStep()（写 t_task_step）
                                    |     ↓
                                    |   检查 cancel_requested 标志
                                    |-- 循环结束
                                    ↓
                              TaskService.updateStatus(SUCCEEDED)
```

---

## 10. 模块间依赖规则

### 10.1 允许的依赖关系

```
auth ──────────────────→ user（查询用户信息用于认证）
chat ──────────────────→ knowledge（RAG 检索）
agent ─────────────────→ chat（使用对话能力）
agent ─────────────────→ tool（工具调用）
agent ─────────────────→ knowledge（知识检索）
task ──────────────────→ agent（提交 Agent 任务）
所有模块 ──────────────→ common（工具类、异常、Result）
所有模块 ──────────────→ ai（通过 Port 接口调用 AI 能力）
```

### 10.2 禁止的依赖关系

```
❌ chat → user（不允许对话模块直接查用户表，改为通过 CurrentUserUtils 获取用户 ID）
❌ knowledge → chat（知识库不了解"对话"概念）
❌ tool → agent（工具不了解"Agent"概念）
❌ 任何模块 → 另一模块的 Mapper（跨模块只能调用 Service 接口，不能访问对方的 Mapper）
❌ 任何模块直接 import SpringAI 或 LangChain4j 的类（必须通过 Port 接口）
```

### 10.3 跨模块调用方式

```java
// 正确：chat 模块调用 knowledge 模块时，依赖 KnowledgeRetrievalService 接口
@Service
@RequiredArgsConstructor
public class ChatServiceImpl implements ChatService {

    private final KnowledgeRetrievalService retrievalService;  // 依赖 Service 接口

    private String buildRagContext(Long kbId, String question) {
        List<RelevantChunk> chunks = retrievalService.search(kbId, question, 5);
        return chunks.stream()
            .map(RelevantChunk::getContent)
            .collect(Collectors.joining("\n\n"));
    }
}

// 错误：chat 模块直接操作 knowledge 模块的 Mapper
// ❌ private final DocumentMapper documentMapper;  // 禁止！
```

---

## 11. 配置管理规范

### 11.1 application.yml 结构

```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/agenthub
    username: ${DB_USERNAME:root}
    password: ${DB_PASSWORD:root}
  data:
    redis:
      host: ${REDIS_HOST:localhost}
      port: 6379

# 自定义配置，统一放在 agenthub 命名空间下
agenthub:
  ai:
    provider: spring-ai          # spring-ai 或 langchain4j
    model: gpt-4o
    api-key: ${OPENAI_API_KEY}   # 敏感信息走环境变量
    base-url: https://api.openai.com/v1

  jwt:
    secret: ${JWT_SECRET}
    access-token-expiry: 7200    # 秒，2小时
    refresh-token-expiry: 604800  # 秒，7天

  vector-store:
    provider: pgvector           # pgvector 或 milvus
    collection-prefix: agenthub_

  task:
    thread-pool:
      core-size: 4
      max-size: 10
      queue-capacity: 100

  rag:
    chunk-size: 512
    chunk-overlap: 50
    top-k: 5
```

### 11.2 多环境配置

```
application.yml           # 公共配置
application-dev.yml       # 开发环境（本地数据库、测试 Key）
application-prod.yml      # 生产环境（走环境变量）
```

---

## 12. 开发阶段建议

### 12.1 推荐顺序

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| 第 1 阶段 | 搭建基础骨架（Spring Boot + MyBatis-Plus + Redis + 统一响应）| 2-3 天 |
| 第 2 阶段 | 用户认证 + 用户管理模块（打通完整 CRUD 链路）| 1 周 |
| 第 3 阶段 | 接入 Spring AI，实现对话引擎 + SSE 流式| 1-2 周 |
| 第 4 阶段 | 工具调用模块（3-4 个工具），理解 Function Calling| 1 周 |
| 第 5 阶段 | 知识库模块，完整 RAG 流水线 | 2 周 |
| 第 6 阶段 | Agent 编排模块（ReAct）| 1-2 周 |
| 第 7 阶段 | 任务管理模块，补齐异步任务链路 | 1 周 |
| 第 8 阶段 | 切换 LangChain4j 实现，体会适配层价值 | 3-5 天 |

### 12.2 先用 Spring AI，再换 LangChain4j

建议先用 Spring AI 跑通所有模块，因为它的 Spring 生态集成度高，上手最快。全部功能稳定后，再切换到 LangChain4j 实现，对比两套框架在 RAG 和 Agent 编排上的 API 设计差异，这比单独学任何一个框架都有收获。

### 12.3 Docker Compose 本地环境

```yaml
# docker-compose.yml
services:
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_DATABASE: agenthub
      MYSQL_ROOT_PASSWORD: root
    ports:
      - "3306:3306"

  postgres:          # 用于 pgvector
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: agenthub_vector
      POSTGRES_PASSWORD: root
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    ports:
      - "6379:6379"
```

一条命令启动所有依赖：`docker compose up -d`

---

*本文档随项目演进持续更新。遇到设计取舍时，优先遵循「简单可读」原则，过度设计是学习阶段最常见的陷阱。*
