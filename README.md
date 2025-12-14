# BioRAG
### : 실시간 PubMed 기반 질병-유전자 RAG 시스템

BioRAG은 최신 PubMed 논문을 기반으로 질병–유전자 관계를 의미 기반으로 검색하고, 근거 문장과 함께 LLM 기반의 응답을 제공하는 생물의학 특화 RAG 시스템이다. 기존 질병–유전자 데이터베이스가 가지는 정적 구조와 키워드 중심 검색의 한계를 극복하고자, 실시간 문헌 검색과 semantic search를 결합한 파이프라인을 설계하였다.
본 시스템은 HGNC 완전 사전과 SciSpaCy BC5CDR 모델을 결합한 entity normalization을 통해, 생물의학 도메인에서 빈번하게 발생하는 유전자 명명법 다양성, 동의어 및 약어 문제를 처리한다. 이후 문장 단위 임베딩과 벡터 검색을 통해 관련 근거 문장을 추출하고, 이를 기반으로 LLM이 질병–유전자 관계에 대한 설명을 생성한다.
BioRAG은 최고 성능 달성보다는, 생물의학 도메인에서 RAG 시스템을 실제로 작동시키기 위해 필요한 데이터 수집, 전처리, 엔티티 정규화, 실시간 검색 파이프라인의 기술적 난이도를 실증적으로 탐구하는 데 목적이 있다. 향후 하드웨어 및 데이터 확장을 통해 연구 생산성과 답변 신뢰성을 높이는 방향으로 확장 가능하다.

## 주요 기능

- **실시간 PubMed 검색** (Entrez API)
- **생물의학 특화 NER** (SciSpaCy BC5CDR + HGNC 사전)
- **Semantic Search** (SentenceTransformer + FAISS)
- **RAG 답변** (MedAlpaca-7B)
- **gene→disease / disease→gene** 양방향 지원

## 디렉토리 구조
```
BioRAG/
├── src/                    # 메인 모듈
├── test/                   # 실시간, 배치 테스트
├── data/                   # 데이터
├── docs/                   # proposal, report 등 문서
└── README.md
```

## 환경 설정
- **Python**: 3.10.x (3.11+에서 scispacy 호환성 이슈)
- **OS**: macOS M2
- **device**: CPU 사용
