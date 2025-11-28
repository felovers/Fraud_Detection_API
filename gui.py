import sys
import os
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
df_full_results = pd.DataFrame()                             
current_cm = None                                               
report_dict_for_export = {}
scalar_metrics_for_export = {}

# --- FUNÇÕES AUXILIARES DE INTERFACE ---

def log_to_screen(message):
    """Escreve uma mensagem na caixa de texto com o horário atual."""
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        report_text.config(state=tk.NORMAL)
        report_text.insert(tk.END, formatted_message)
        report_text.see(tk.END) # Rola automaticamente para o final
        report_text.config(state=tk.DISABLED)
    except:
        pass 

# --- FUNÇÕES LÓGICAS ---

def run_evaluation_thread():
    """
    Função principal que roda em segundo plano.
    """
    try:
        # --- 1. Carregar o Dataset ---
        log_to_screen("Aguardando seleção do arquivo de dados...")
        filepath = filedialog.askopenfilename(
            title="Selecione o Arquivo Desejado",
            filetypes=[("Arquivos CSV", "*.csv")]
        )
        if not filepath:
            status_var.set("Status: Ocioso")
            log_to_screen("Seleção cancelada pelo usuário.")
            return 
            
        status_var.set("Status: Carregando dataset...")
        log_to_screen(f"Carregando arquivo: {filepath.split('/')[-1]}...")
        
        try:
            df_original = pd.read_csv(filepath)
            
            # Validação
            if 'Class' not in df_original.columns:
                messagebox.showerror("Erro", "Dataset inválido. Coluna 'Class' não encontrada.")
                status_var.set("Status: Ocioso")
                return

            log_to_screen(f"Dataset carregado com sucesso")

        except Exception as e:
            messagebox.showerror("Erro ao ler CSV", f"Não foi possível ler o arquivo: {e}")
            status_var.set("Status: Ocioso")
            return

        # --- 2. Separar e Fazer o Split (Anti-Data Leakage) ---
        status_var.set("Status: Iniciando Avaliação...")
        log_to_screen("--------------------------------------------------")
        log_to_screen("INICIANDO PROCESSO DE AVALIAÇÃO") 
        
        X = df_original.drop("Class", axis=1)
        y = df_original["Class"]

        # Split 80/20
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        true_labels = y_test.copy()
        features_df = X_test.copy()
        
        log_to_screen(f"Enviando {len(features_df):,} registros para a API.")
        log_to_screen("--------------------------------------------------")

        # --- 3. Dividir em Chunks e Processar ---
        chunk_size = 10000
        all_predictions = [] 
        
        chunks = np.array_split(features_df, (len(features_df) // chunk_size) + 1)
        num_chunks = len(chunks)
        
        # Configura a barra de progresso
        progress_bar['maximum'] = num_chunks
        progress_bar['value'] = 0

        for i, chunk_df in enumerate(chunks):
            if chunk_df.empty:
                continue 
            
            current_step = i + 1
            status_var.set(f"Status: Enviando Lote {current_step}/{num_chunks} para a nuvem...")
            
            try:
                csv_in_memory = chunk_df.to_csv(index=False)
                files_payload = {'file': ('chunk.csv', csv_in_memory, 'text/csv')}

                # Chama a API
                response = requests.post(API_URL, files=files_payload)
                response.raise_for_status() 
                
                response_data = response.json()
                if 'predictions' not in response_data:
                    raise KeyError("JSON inválido da API.")
                    
                chunk_predictions = response_data['predictions']
                all_predictions.extend(chunk_predictions) 
                
                # Atualiza barra de progresso
                progress_bar['value'] = current_step
                
                # Log a cada requisição bem sucedida
                log_to_screen(f"✔ Lote {current_step}/{num_chunks}: API processou {len(chunk_predictions)} transações.")

            except Exception as e:
                log_to_screen(f"❌ Erro no Lote {current_step}: {str(e)}")
                messagebox.showerror("Erro de API", str(e))
                status_var.set("Status: Erro de API")
                return 

        # --- 4. Geração de Relatórios ---
        status_var.set("Status: Calculando métricas finais...")
        log_to_screen("--------------------------------------------------")
        log_to_screen("Todos os lotes processados. Consolidando resultados...")
        
        predictions = all_predictions
        
        if len(predictions) != len(true_labels):
             messagebox.showerror("Erro", "Discrepância no número de predições recebidas.")
             return

        # 4.1. Métricas
        report_dict = classification_report(true_labels, predictions, target_names=['Normal', 'Fraude'], zero_division=0, output_dict=True)
        
        try:
            auc_score = roc_auc_score(true_labels, predictions)
        except ValueError:
            auc_score = 0.0
        
        # Prepara para exportação Excel
        global report_dict_for_export, scalar_metrics_for_export
        scalar_metrics_for_export = {
            'accuracy': report_dict.pop('accuracy'),
            'auc-roc_score': auc_score
        }
        report_dict_for_export = report_dict 
        
        # Gera texto para exibição
        report_str = classification_report(true_labels, predictions, target_names=['Normal', 'Fraude'], zero_division=0)
        report_str += f"\n\nAUC-ROC Score: {auc_score:.6f}\n" 
        
        log_to_screen("Relatório Final Gerado:")
        log_to_screen("\n" + report_str)
        
        # 4.2. Matriz e DataFrame COMPLETO
        global current_cm, df_full_results
        current_cm = confusion_matrix(true_labels, predictions)
        
        df_test_results = X_test.copy()
        df_test_results['Class_Real'] = y_test
        df_test_results['prediction'] = predictions
        df_full_results = df_test_results 
        
        detected_count = len(df_test_results[df_test_results['prediction'] == 1])
        
        btn_show_matrix.config(state=tk.NORMAL)
        btn_export_report.config(state=tk.NORMAL)
        
        # --- FINALIZAÇÃO ---
        status_var.set(f"Avaliação concluída. {detected_count} fraudes detectadas.")
        log_to_screen(f"PROCESSO FINALIZADO COM SUCESSO.")

    except Exception as e:
        messagebox.showerror("Erro Geral", str(e))
        status_var.set("Status: Erro")
        log_to_screen(f"ERRO FATAL: {str(e)}")

def start_evaluation():
    btn_run.config(state=tk.DISABLED)
    btn_show_matrix.config(state=tk.DISABLED)
    btn_export_report.config(state=tk.DISABLED)
    report_text.config(state=tk.NORMAL)
    report_text.delete('1.0', tk.END) 
    report_text.config(state=tk.DISABLED)
    progress_bar['value'] = 0 
    
    eval_thread = Thread(target=run_evaluation_thread, daemon=True)
    eval_thread.start()
    root.after(100, check_thread, eval_thread)

def check_thread(thread):
    if not thread.is_alive():
        btn_run.config(state=tk.NORMAL)
    else:
        root.after(100, check_thread, thread)

def show_confusion_matrix():
    if current_cm is None: return
    matrix_window = tk.Toplevel(root)
    matrix_window.title("Matriz de Confusão")
    tn, fp, fn, tp = current_cm.ravel()
    labels = [[f'VN (Normal)\n{tn}', f'FP (Alarme Falso)\n{fp}'], [f'FN (Perda)\n{fn}', f'VP (Fraude Real)\n{tp}']]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(current_cm, annot=labels, fmt="", cmap='Blues', xticklabels=['Prev. Normal', 'Prev. Fraude'], yticklabels=['Real Normal', 'Real Fraude'], annot_kws={"size": 12})
    ax.set_title('Matriz de Confusão')
    ax.set_xlabel('Predição do Modelo')
    ax.set_ylabel('Valor Real')
    canvas = FigureCanvasTkAgg(fig, master=matrix_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def export_detected_frauds_report():
    """Gera o Excel agora com 4 ABAS, incluindo Falsos Negativos."""
    global df_full_results, report_dict_for_export, scalar_metrics_for_export
    if not report_dict_for_export: return
    try:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        suggested_filename = f"Relatorio_Avaliacao_Fraude_{timestamp}.xlsx"
        filepath = filedialog.asksaveasfilename(title="Salvar Relatório Excel", initialfile=suggested_filename, defaultextension=".xlsx", filetypes=[("Arquivos Excel", "*.xlsx")])
        if filepath:
            # 1. Resumo
            df_report = pd.DataFrame(report_dict_for_export).transpose().reset_index().rename(columns={'index': 'Métrica'})
            df_scalars = pd.DataFrame.from_dict(scalar_metrics_for_export, orient='index', columns=['Score']).reset_index().rename(columns={'index': 'Métrica'})
            
            # 2. Filtragem dos dados (VP, FP, FN)
            # VP: Era Fraude (1) e Modelo disse Fraude (1)
            df_tp = df_full_results[(df_full_results['Class_Real'] == 1) & (df_full_results['prediction'] == 1)]
            
            # FP: Era Normal (0) e Modelo disse Fraude (1)
            df_fp = df_full_results[(df_full_results['Class_Real'] == 0) & (df_full_results['prediction'] == 1)]
            
            # FN (NOVO): Era Fraude (1) e Modelo disse Normal (0) -> A Fraude que passou
            df_fn = df_full_results[(df_full_results['Class_Real'] == 1) & (df_full_results['prediction'] == 0)]

            cols_drop = ['Class_Real', 'prediction']
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Aba 1
                df_report.to_excel(writer, sheet_name='Resumo', index=False, startrow=0)
                df_scalars.to_excel(writer, sheet_name='Resumo', index=False, startrow=len(df_report) + 2)
                
                # Aba 2: Sucessos
                df_tp.drop(columns=cols_drop, errors='ignore').to_excel(writer, sheet_name='Fraudes Reais (VP)', index=False)
                
                # Aba 3: Alarmes Falsos
                df_fp.drop(columns=cols_drop, errors='ignore').to_excel(writer, sheet_name='Alarmes Falsos (FP)', index=False)
                
                # Aba 4: Prejuízos (Fraudes Não Detectadas)
                df_fn.drop(columns=cols_drop, errors='ignore').to_excel(writer, sheet_name='Fraudes Nao Detectadas (FN)', index=False)

            messagebox.showinfo("Sucesso", f"Relatório salvo com 4 abas em:\n{filepath}")
    except Exception as e:
        messagebox.showerror("Erro", str(e))

# --- CONFIGURAÇÃO DA JANELA PRINCIPAL (GUI) ---
root = tk.Tk()
root.title("Sistema de Detecção de Fraudes") 
root.geometry("700x550") 

main_frame = ttk.Frame(root, padding="15")
main_frame.pack(fill=tk.BOTH, expand=True)

# 2. Frame de Botões
button_frame = ttk.LabelFrame(main_frame, text="Controles", padding="10")
button_frame.pack(fill=tk.X, pady=5)

btn_run = ttk.Button(button_frame, text="▶ Iniciar Avaliação (Carregar Dataset)", command=start_evaluation)
btn_run.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

btn_show_matrix = ttk.Button(button_frame, text="📊 Ver Matriz de Confusão", state=tk.DISABLED, command=show_confusion_matrix)
btn_show_matrix.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

btn_export_report = ttk.Button(button_frame, text="💾 Exportar Relatório Excel", state=tk.DISABLED, command=export_detected_frauds_report)
btn_export_report.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

# 3. Frame do Relatório (LOG)
report_frame = ttk.LabelFrame(main_frame, text="Log de Execução e Resultados", padding="10")
report_frame.pack(fill=tk.BOTH, expand=True, pady=10)

report_text = scrolledtext.ScrolledText(report_frame, state=tk.DISABLED, wrap=tk.WORD, font=("Consolas", 9))
report_text.pack(fill=tk.BOTH, expand=True)

# 4. Barra de Progresso
progress_bar = ttk.Progressbar(main_frame, orient='horizontal', mode='determinate')
progress_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

# 5. Barra de Status
status_var = tk.StringVar(value="Status: Sistema pronto. Aguardando dataset.")
status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W, padding="5", background="#ecf0f1")
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# --- FUNÇÃO DE FECHAMENTO SEGURO ---
def on_closing():
    """Força o encerramento de todas as threads e do processo ao fechar a janela."""
    root.destroy()
    os._exit(0)

# Vincula o botão "X" da janela à função on_closing
root.protocol("WM_DELETE_WINDOW", on_closing)

# --- INICIAR APLICAÇÃO ---
root.mainloop()