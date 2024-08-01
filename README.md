# ChatBot Processo Seletivo NeuralMind
Chatbot para responder dúvidas acerca do Vestibular Unicamp 2025 feito como parte do processo seletivo de estágio da NeuralMind jul/2024.

## Pré-requisitos

Para execução do script, previamente deve-se ter instalado:
 - Python 3.10 e Pip 22.3 ou superior
 - [Pipenv](https://pipenv.pypa.io/en/latest/#install-pipenv-today) 2022.12.19 ou mais recente

## Como utilizar
Primeiro, clone o repositório em sua máquina. Então, para começar, na raíz do projeto, execute:
```bash
pipenv install --python 3.10
```

Execute os comandos dentro do shell:
```bash
pipenv shell
```

Finalmente, para inicializar o chatbot, execute o seguinte comando:
```bash
streamlit run src/main.py
```

Uma janela será gerada em seu browser com a interface do chatbot. Digite sua pergunta no campo designado, clique no botão **'Gerar resposta'** e aguarde sua resposta!

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

### Uso do ChatGPT

Utilizei o ChatGPT para obter um início para o código da cadeia de RAG visto em src/main.py, mas diversas vezes as soluções dadas se demonstraram insuficientes, então busquei outras na internet (StackOverflow e documentação do LangChain principalmente) e depois voltei ao ChatGPT apenas para corrigir erros em minha implementação.

Também o utilizei para escrever um modelo base para esse relatório e depois alterei e inclui o que achei necessário.


## Resultados
#### BLEU (Bilingual Evaluation Understudy)
- **BLEU Score** :0.2017.

Esta métrica mede a precisão da sobreposição de n-gramas entre a resposta gerada pelo modelo e uma ou mais respostas de referência. Uma pontuação de 0.2017 indica uma correspondência moderada. BLEU é particularmente útil para avaliar traduções automáticas e respostas curtas.

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **ROUGE-1**: Mede a sobreposição de unigramas (palavras individuais) entre a resposta gerada e a resposta de referência.
  - **Recall (r)**: 0.7441. Indica que o modelo captura 74.41% das palavras relevantes.
  - **Precision (p)**: 0.1709. Reflete que 17.09% das palavras geradas pelo modelo são relevantes.
  - **F1 Score (f)**: 0.2661. É a média harmônica de precisão e recall, balanceando ambos.

- **ROUGE-2**: Mede a sobreposição de bigramas (pares de palavras) entre a resposta gerada e a resposta de referência.
  - **Recall (r)**: 0.6261. Mostra que o modelo captura 62.61% dos bigramas relevantes.
  - **Precision (p)**: 0.1141. Indica que 11.41% dos bigramas gerados são relevantes.
  - **F1 Score (f)**: 0.1813. Média harmônica de precisão e recall para bigramas.

- **ROUGE-L**: Mede a sobreposição da subsequência mais longa (longest common subsequence) entre a resposta gerada e a de referência, refletindo a estrutura sequencial do texto.
  - **Recall (r)**: 0.7403. Indica que o modelo captura 74.03% da estrutura sequencial relevante.
  - **Precision (p)**: 0.1693. Reflete que 16.93% da sequência gerada é relevante.
  - **F1 Score (f)**: 0.2639. Média harmônica de precisão e recall para a estrutura sequencial.
