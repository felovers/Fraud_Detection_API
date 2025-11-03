import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread

# --- VARIÁVEIS GLOBAIS ---
API_URL = "https://fraud-detection-api-7ehe.onrender.com/predict" # URL da API
df_detected_frauds = pd.DataFrame()     
current_cm = None                       
report_dict_for_export = {}
scalar_metrics_for_export = {}

# Lista de colunas que o modelo espera, na ordem correta
EXPECTED_COLUMNS = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

# --- FUNÇÕES LÓGICAS ---

def run_evaluation_thread():
    """
    Função 'wrapper' que lê o CSV, faz o split 80/20,
    e envia APENAS o conjunto de teste (20%) em chunks
    para a API.
    """
    try:
        # --- 1. Carregar o Dataset ---
        filepath = filedialog.askopenfilename(
            title="Selecione o Dataset COMPLETO (creditcard.csv)",
            filetypes=[("Arquivos CSV", "*.csv")]
        )
        if not filepath:
            status_var.set("Status: Ocioso")
            return # Usuário cancelou
            
        status_var.set("Status: Carregando e dividindo o dataset...")
        
        try:
            df_original = pd.read_csv(filepath)
            
            # Validação: Checa se a coluna de gabarito 'Class' existe
            if 'Class' not in df_original.columns:
                messagebox.showerror("Erro", "Dataset inválido. Coluna 'Class' (gabarito) não encontrada.")
                status_var.set("Status: Ocioso")
                return

        except Exception as e:
            messagebox.showerror("Erro ao ler CSV", f"Não foi possível ler o arquivo: {e}")
            status_var.set("Status: Ocioso")
            return

        # --- 2. Separar e Fazer o Split (A CORREÇÃO DO BUG) ---
        X = df_original.drop("Class", axis=1)
        y = df_original["Class"]

        # Replicando o split exato do script de teste (e do treino do modelo)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # O gabarito é SÓ o y_test
        true_labels = y_test.copy()
        
        # As features são SÓ o X_test
        features_df = X_test.copy()
        
        print(f"[INFO] Dataset dividido. Enviando {len(features_df)} linhas de teste para a API...")

        # --- 3. Dividir em Chunks e Processar ---
        chunk_size = 10000
        all_predictions = [] 
        
        chunks = np.array_split(features_df, (len(features_df) // chunk_size) + 1)
        num_chunks = len(chunks)

        for i, chunk_df in enumerate(chunks):
            if chunk_df.empty:
                continue 
            
            status_var.set(f"Status: Processando chunk de teste {i+1} de {num_chunks}...")
            
            try:
                csv_in_memory = chunk_df.to_csv(index=False)
                
                files_payload = {
                    'file': ('chunk.csv', csv_in_memory, 'text/csv')
                }

                response = requests.post(API_URL, files=files_payload)
                response.raise_for_status() 
                
                response_data = response.json()
                if 'predictions' not in response_data:
                    raise KeyError("A resposta da API não contém a chave 'predictions'.")
                    
                chunk_predictions = response_data['predictions']
                all_predictions.extend(chunk_predictions) 

            except requests.exceptions.HTTPError as http_err:
                messagebox.showerror(f"Erro de API (Chunk {i+1})", f"A API retornou um erro:\n{http_err.response.text}")
                status_var.set("Status: Erro de API")
                return 
            except requests.exceptions.RequestException as e:
                messagebox.showerror("Erro de Conexão", f"Não foi possível conectar à API em {API_URL}\nErro: {e}")
                status_var.set("Status: Erro de API")
                return 
            except KeyError as e:
                messagebox.showerror("Erro de Resposta", f"Erro ao processar resposta da API: {e}")
                status_var.set("Status: Erro de API")
                return 

        # --- 4. Geração de Relatórios (Pós-Loop) ---
        status_var.set("Status: API processou. Gerando relatórios...")
        
        predictions = all_predictions
        
        if len(predictions) != len(true_labels):
             messagebox.showerror("Erro de Resposta", 
                                  f"Erro de contagem: O CSV de teste tinha {len(true_labels)} linhas, "
                                  f"mas a API retornou um total de {len(predictions)} predições.")
             status_var.set("Status: Erro de resposta")
             return

        # 4.1. Relatório de Classificação e AUC-ROC
        report_dict = classification_report(true_labels, predictions, target_names=['Normal', 'Fraude'], zero_division=0, output_dict=True)
        
        try:
            auc_score = roc_auc_score(true_labels, predictions)
        except ValueError:
            auc_score = 0.0 # ou None
        
        global report_dict_for_export, scalar_metrics_for_export
        
        scalar_metrics_for_export = {
            'accuracy': report_dict.pop('accuracy'), # Remove 'accuracy' do dict principal
            'auc-roc_score': auc_score
        }
        report_dict_for_export = report_dict 
        
        report_str = classification_report(true_labels, predictions, target_names=['Normal', 'Fraude'], zero_division=0)
        report_str += f"\n\nAUC-ROC Score: {auc_score:.6f}\n" # Adiciona o AUC
        
        report_text.config(state=tk.NORMAL)
        report_text.delete('1.0', tk.END)
        report_text.insert(tk.END, report_str) # Insere o texto na GUI
        report_text.config(state=tk.DISABLED)
        
        # 4.2. Matriz de Confusão
        global current_cm
        current_cm = confusion_matrix(true_labels, predictions)
        
        # 4.3. Relatório de Fraudes Detectadas (CSV)
        global df_detected_frauds
        # Adiciona as predições ao DataFrame DE TESTE (X_test)
        df_test_results = X_test.copy()
        df_test_results['Class_Real'] = y_test
        df_test_results['prediction'] = predictions
        
        df_detected_frauds = df_test_results[df_test_results['prediction'] == 1]
        
        # Habilita os botões de relatório
        btn_show_matrix.config(state=tk.NORMAL)
        btn_export_report.config(state=tk.NORMAL)
        
        status_var.set(f"Status: Avaliação concluída. ({len(df_detected_frauds)} fraudes detectadas)")

    except Exception as e:
        messagebox.showerror("Erro Inesperado (Geral)", str(e))
        status_var.set("Status: Erro")

def start_evaluation():
    """Inicia a avaliação em uma nova thread para não bloquear a GUI."""
    # Desabilita botões para evitar cliques duplos
    btn_run.config(state=tk.DISABLED)
    btn_show_matrix.config(state=tk.DISABLED)
    btn_export_report.config(state=tk.DISABLED)
    
    # Inicia a thread
    eval_thread = Thread(target=run_evaluation_thread, daemon=True)
    eval_thread.start()
    
    # Reabilita o botão de rodar após a conclusão (a thread faz isso)
    # Aqui, garantimos que o botão de "rodar" seja reabilitado se a thread falhar
    root.after(100, check_thread, eval_thread)

def check_thread(thread):
    """Verifica se a thread terminou e reabilita o botão 'Rodar'."""
    if not thread.is_alive():
        btn_run.config(state=tk.NORMAL)
    else:
        root.after(100, check_thread, thread)

def show_confusion_matrix():
    """
    Cria uma nova janela (Toplevel) para exibir a Matriz de Confusão.
    [Versão com rótulos TN, FP, FN, TP]
    """
    if current_cm is None:
        messagebox.showwarning("Aviso", "Nenhum dado de matriz de confusão. Rode a avaliação primeiro.")
        return
        
    matrix_window = tk.Toplevel(root)
    matrix_window.title("Matriz de Confusão")
    
    # Extrai os valores de TN, FP, FN, TP
    # ravel() achata a matriz 2x2 para [TN, FP, FN, TP]
    tn, fp, fn, tp = current_cm.ravel()
    
    # Cria os rótulos de texto
    labels = [
        [f'Verdadeiro Negativo (TN)\n{tn}', f'Falso Positivo (FP)\n{fp}'],
        [f'Falso Negativo (FN)\n{fn}', f'Verdadeiro Positivo (TP)\n{tp}']
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(current_cm, annot=labels, fmt="", cmap='Blues',
                xticklabels=['Prev. Normal', 'Prev. Fraude'],
                yticklabels=['Real Normal', 'Real Fraude'],
                annot_kws={"size": 12})
                
    ax.set_title('Matriz de Confusão')
    ax.set_xlabel('Predição do Modelo')
    ax.set_ylabel('Valor Real')
    
    canvas = FigureCanvasTkAgg(fig, master=matrix_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def export_detected_frauds_report():
    """
    Salva um RELATÓRIO EXCEL multi-abas (xlsx) contendo:
    1. Resumo da Avaliação (COM COLUNAS CORRETAS)
    2. Lista de Fraudes Reais (Verdadeiros Positivos)
    3. Lista de Alarmes Falsos (Falsos Positivos)
    """
    
    global df_detected_frauds, report_dict_for_export, scalar_metrics_for_export

    if not report_dict_for_export: # Verifica se o dicionário não está vazio
        messagebox.showwarning("Aviso", "Nenhum dado para exportar.\nRode a avaliação primeiro.")
        return
        
    try:
        # --- 1. Gera o nome do arquivo ---
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        suggested_filename = f"Relatorio_Avaliacao_Fraude_{timestamp}.xlsx"

        # --- 2. Abre a janela "Salvar como..." ---
        filepath = filedialog.asksaveasfilename(
            title="Salvar Relatório de Avaliação",
            initialfile=suggested_filename,
            defaultextension=".xlsx",
            filetypes=[("Arquivos Excel", "*.xlsx")]
        )
        
        if filepath:
            
            # --- 3. Prepara os DataFrames para as abas ---
            
            # --- [INÍCIO DA CORREÇÃO] ---
            # Prepara a aba de Resumo (AGORA EM COLUNAS)
            
            # Parte 1: O relatório principal (Precision, Recall, etc.)
            df_report = pd.DataFrame(report_dict_for_export).transpose()
            df_report.reset_index(inplace=True) # Move 'Normal', 'Fraude' de index para coluna
            df_report = df_report.rename(columns={'index': 'Métrica'})
            
            # Parte 2: Os scores únicos (Accuracy, AUC)
            df_scalars = pd.DataFrame.from_dict(
                scalar_metrics_for_export, 
                orient='index', 
                columns=['Score']
            )
            df_scalars.reset_index(inplace=True)
            df_scalars = df_scalars.rename(columns={'index': 'Métrica'})
            # --- [FIM DA CORREÇÃO] ---
            
            
            # Prepara as outras abas (VP e FP) - (Este código já estava correto)
            df_tp = df_detected_frauds[df_detected_frauds['Class_Real'] == 1]
            df_fp = df_detected_frauds[df_detected_frauds['Class_Real'] == 0]
            
            columns_to_drop = ['Class_Real', 'prediction']
            df_tp_export = df_tp.drop(columns=columns_to_drop, errors='ignore')
            df_fp_export = df_fp.drop(columns=columns_to_drop, errors='ignore')
            
            
            # --- 4. Cria o arquivo Excel com múltiplas abas ---
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                
                # Aba 1: Resumo (com 2 tabelas)
                df_report.to_excel(writer, sheet_name='Resumo da Avaliacao', index=False, startrow=0)
                
                # Adiciona a segunda tabela (scalars) com espaço
                # Começa a escrever na linha (len(df_report) + 2)
                df_scalars.to_excel(writer, sheet_name='Resumo da Avaliacao', index=False, startrow=len(df_report) + 2)
                
                # Aba 2: Verdadeiros Positivos
                df_tp_export.to_excel(writer, sheet_name='Fraudes Reais (VP)', index=False)
                
                # Aba 3: Falsos Positivos
                df_fp_export.to_excel(writer, sheet_name='Alarmes Falsos (FP)', index=False)
            
            messagebox.showinfo("Sucesso", f"Relatório Excel salvo com sucesso em:\n{filepath}")
            
    except ImportError:
         messagebox.showerror("Erro de Biblioteca", 
                              "A biblioteca 'openpyxl' é necessária para salvar arquivos Excel.\n"
                              "Por favor, instale-a com:\n\npip install openpyxl")
    except Exception as e:
        messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar o arquivo:\n{e}")

# --- CONFIGURAÇÃO DA JANELA PRINCIPAL (GUI) ---
root = tk.Tk()
root.title("Avaliador da API de Detecção de Fraude")
root.geometry("600x500")

main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# --- Frame de Botões (Topo) ---
button_frame = ttk.Frame(main_frame)
button_frame.pack(fill=tk.X, pady=5)

btn_run = ttk.Button(button_frame, text="Carregar Dataset e Rodar Avaliação", command=start_evaluation)
btn_run.pack(side=tk.LEFT, padx=5)

btn_show_matrix = ttk.Button(button_frame, text="Ver Matriz de Confusão", state=tk.DISABLED, command=show_confusion_matrix)
btn_show_matrix.pack(side=tk.LEFT, padx=5)

btn_export_report = ttk.Button(button_frame, text="Exportar Fraudes Detectadas (CSV)", state=tk.DISABLED, command=export_detected_frauds_report)
btn_export_report.pack(side=tk.LEFT, padx=5)

# --- Frame do Relatório (Meio) ---
report_frame = ttk.LabelFrame(main_frame, text="Relatório de Classificação", padding="10")
report_frame.pack(fill=tk.BOTH, expand=True, pady=10)

report_text = scrolledtext.ScrolledText(report_frame, state=tk.DISABLED, wrap=tk.WORD, height=15)
report_text.pack(fill=tk.BOTH, expand=True)

# --- Barra de Status (Baixo) ---
status_var = tk.StringVar(value="Status: Ocioso.")
status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W, padding="5")
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# --- INICIAR APLICAÇÃO ---
root.mainloop()