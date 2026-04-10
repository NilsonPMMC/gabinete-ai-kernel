import requests
import time

# Atenção: Seus logs mostraram porta 8000, então vamos testar nela.
BASE_URL = "http://localhost:8001"

def testar_embeddings():
    print("\n--- Teste 1: Gerar Embeddings (Vetorização) ---")
    payload = {
        "texts": [
            "O SIGA é um sistema de gestão pública.",
            "A prefeitura de Mogi das Cruzes fica em SP."
        ]
    }
    
    inicio = time.time()
    try:
        response = requests.post(f"{BASE_URL}/v1/embeddings", json=payload)
        tempo = time.time() - inicio
        
        if response.status_code == 200:
            dados = response.json()
            # O formato de retorno depende de como você fez o main.py.
            # Geralmente retorna {"embeddings": [[...], [...]]} ou lista direta.
            embeddings = dados.get('embeddings', dados) 
            
            print(f"✅ Sucesso! Resposta em {tempo:.2f}s")
            if isinstance(embeddings, list) and len(embeddings) > 0:
                print(f"   Qtd textos processados: {len(embeddings)}")
                print(f"   Tamanho do vetor (dimensões): {len(embeddings[0])} (Esperado: 1024)")
            else:
                print("   ⚠️ Formato de resposta inesperado:", dados)
        else:
            print(f"❌ Erro {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Falha na conexão: {e}")

def testar_similaridade():
    print("\n--- Teste 2: Verificar Duplicidade (Similaridade) ---")
    payload = {
        "target_text": "João da Silva",
        "candidates": [
            "Joao Silva",           # Quase igual (deve ser alto)
            "Maria Oliveira",       # Diferente (deve ser baixo)
            "Prefeitura Municipal"  # Nada a ver (deve ser muito baixo)
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/v1/similarity", json=payload)
        
        if response.status_code == 200:
            dados = response.json()
            print("✅ Resultado da Comparação:")
            # Ajuste conforme o retorno da sua API (ex: lista de dicts ou dict com scores)
            print(dados)
        else:
            print(f"❌ Erro {response.status_code}: {response.text}")

    except Exception as e:
        print(f"❌ Falha na conexão: {e}")

if __name__ == "__main__":
    print(f"Conectando em {BASE_URL}...")
    testar_embeddings()
    testar_similaridade()