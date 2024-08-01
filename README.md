# ChatBot PS NeuralMind
Chatbot para responder dúvidas acerca do Vestibular Unicamp 2025 feito como parte do processo seletivo de estágio da NeuralMind jul/2024.

## Pré-requisitos

Para execução do script, previamente deve-se ter instalado:
 - Python 3.10 e Pip 22.3 ou superior
 - [Pipenv](https://pipenv.pypa.io/en/latest/#install-pipenv-today) 2022.12.19 ou mais recente

## Como utilizar
Para instalar e gerenciar dependências, usei o Pipenv. 

Para começar, execute:
```bash
pipenv install --python 3.10
```

Para contribuir e desenvolver o projeto, execute:
```bash
pipenv install --python 3.10 --dev
```

Execute os comandos dentro do shell:
```bash
pipenv shell
```

Finalmente, para inicializar o chatbot, execute o seguinte comando:
```bash
streamlit run src/main.py
```

Uma janela será gerada em seu browser com a interface do chatbot. Digite sua pergunta no campo designado, clique no botão 'Gerar resposta' e aguarde sua resposta!

## Como testar

Para testar o chatbot, obtendo métricas de avaliação, você pode executar o arquivo já disponibilizado executando o seguinte comando no terminal:

```bash
python src/test_retriever.py
```

Se quiser gerar seu próprio teste, altere o arquivo data/questions_answers.json (link) como desejar.

## Detalhes da implementação
Carreguei o arquivo PDF da publicação da Resolução GR-029/2024, de 10/07/2024 que "Dispõe sobre o Vestibular Unicamp 2025 para vagas no ensino de Graduação" (link) utilizando PyPDFLoader. Então, o texto do documento é dividido em partes (chunks) menores e gerenciáveis usando RecursiveCharacterTextSplitter.

O modelo de linguagem gpt-4o é inicializado com ChatOpenAI. As partes do texto são transformadas em vetores para facilitar a recuperação de informações utilizando Chroma e OpenAIEmbeddings. Um prompt do sistema é configurado para orientar o chatbot a responder de maneira concisa e específica às perguntas sobre o vestibular, montado com ChatPromptTemplate.

Duas cadeias principais são criadas: uma para combinar documentos e outra para recuperação e resposta (retrieval chain e question answering chain). A interface do chatbot é construída usando Streamlit, permitindo que os usuários insiram suas perguntas sobre o vestibular em um ambiente mais intuitivo e acessível. Ao clicar no botão, a pergunta do usuário é enviada através da cadeia de recuperação e resposta, e a resposta gerada é exibida na interface.

## Resultados

#### BLEU Score
- **BLEU**: 0.2017

#### ROUGE Scores
- **ROUGE-1**:
  - **Recall (r)**: 0.7441
  - **Precision (p)**: 0.1709
  - **F1 Score (f)**: 0.2661

- **ROUGE-2**:
  - **Recall (r)**: 0.6261
  - **Precision (p)**: 0.1141
  - **F1 Score (f)**: 0.1813

- **ROUGE-L**:
  - **Recall (r)**: 0.7403
  - **Precision (p)**: 0.1693
  - **F1 Score (f)**: 0.2639

### Análise dos Resultados
- **BLEU Score**: O valor de 0.2017 indica uma moderada correspondência entre as frases geradas pelo modelo e as de referência, medindo a precisão da sobreposição de n-gramas.
  
- **ROUGE-1**: A pontuação de Recall alta (0.7441) sugere que o modelo captura bem os elementos relevantes, mas a precisão (0.1709) e o F1 Score (0.2661) indicam que há espaço para melhorias na exatidão das respostas.

- **ROUGE-2**: A pontuação de Recall de 0.6261 mostra uma boa captura de bigramas relevantes, porém a precisão (0.1141) e o F1 Score (0.1813) também indicam a necessidade de melhorar a precisão.

- **ROUGE-L**: A pontuação de Recall alta (0.7403) reflete a capacidade do modelo de capturar a estrutura sequencial dos textos, mas a precisão (0.1693) e o F1 Score (0.2639) sugerem que ainda há um desequilíbrio entre a recuperação de informações e a precisão delas.

Essas métricas ajudam a avaliar a qualidade das respostas geradas pelo modelo em relação às de referência, destacando pontos fortes e áreas que necessitam de melhorias.
