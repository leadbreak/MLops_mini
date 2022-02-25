import streamlit as st
import pandas as pd
import numpy as np
import os, time
from glob import glob
import tensorflow as tf
from datetime import datetime
import plotly.graph_objects as go

class st_lwai :

    def __init__(self, ds_option='small', th_value=0.5, PTM_OPTIONS = 'default') :            
        
        if ds_option == 'small':                
            self.DATA_PATH = (os.path.join(os.path.expanduser('~'), 'Documents/in2wise/LwAID/LwALD-raspbian/lwai-learn-ver2/lwai/IMU_data/train/', 'small'))
        elif ds_option == 'normal' :        
            self.DATA_PATH = (os.path.join(os.path.expanduser('~'), 'Documents/in2wise/LwAID/LwALD-raspbian/lwai-learn-ver2/lwai/IMU_data/train/', 'normal'))
       
        
        self.DATA_FILES = glob(f'{self.DATA_PATH}/*.csv')

        self.file_cnt = 0
        self.stop_code = 0

        self.wrong_score = 0
        self.all_count = 0
        self.real_normal = 0
        self.real_abnormal = 0
        self.pred_normal = 0
        self.pred_abnormal = 0
        self.TP_score = 0
        self.TN_score = 0
        self.FN_score = 0
        self.FP_score = 0
        self.accuracy = 1
        self.precision = 1
        self.recall = 1
        self.f1_score = 1 

        self.threshold = th_value

        if PTM_OPTIONS == 'hpt_01' :
            self.model_path = (os.path.join(os.path.expanduser('~'), "Documents\in2wise\st_lwaid\stage02\models", "model_hpt_01.tflite"))
        elif PTM_OPTIONS == 'hpt_02' :
            self.model_path = (os.path.join(os.path.expanduser('~'), "Documents\in2wise\st_lwaid\stage02\models", "model_hpt_02.tflite"))
        elif PTM_OPTIONS == 'default' :
            self.model_path = (os.path.join(os.path.expanduser('~'), "Documents\in2wise\st_lwaid\stage02\models", "default.tflite"))
        elif PTM_OPTIONS == 'stupid_for_test':
            self.model_path = (os.path.join(os.path.expanduser('~'), "Documents\in2wise\st_lwaid\stage02\models", "stupid_for_test.tflite"))
        else :
            self.model_path = PTM_OPTIONS
        self.model_name = self.model_path.split("\\")[-1].replace(".tflite","")

        self.threshold_df = pd.DataFrame()

        self.error_code = 3 # 0 : no problem, 1 : change threshold, 2 : should do retraining, 3 : default

        self.pred_log_path = os.path.join(os.path.expanduser('~'), "Documents\in2wise\st_lwaid\stage02\pred_log", "pred_log.csv")
        self.threshold_df_path = os.path.join(os.path.expanduser('~'), "Documents\in2wise\st_lwaid\stage02\pred_log", f'{str(datetime.today().strftime("%Y%m%d_%H%M"))}_{self.model_name}_errorcode{self.error_code}_threshold_{self.threshold}.csv')



    @st.cache
    def load_data(self):
        
        file = self.DATA_FILES[self.file_cnt]
        # print('file :', file)
        data = pd.read_csv(file, header=None)
        local_time = ":".join(file.split('\\')[-1].split("_")[:-2])
        label_text = file.split("_")[-1].replace(".csv", "")
        if label_text == 'normal' :
            label = 1
        else :
            label = 0

        data.columns = ['accel_x','accel_y','accel_z','gyros_x','gyros_y','gyros_z']        

        return local_time, data, label

    def show_Score(self) :
        # confusion_matrix

        with st1 :
            st.markdown("<h1 style='text-align: center; color: cadetblue;'>SCOREBOARD</h1>", unsafe_allow_html=True)
        with cm1 :
            st.metric(label='TruePositive', value=self.TP_score)
        with cm2 :
            st.metric(label='TrueNegative', value=self.TN_score)
        with cm3 :
            st.metric(label='FalsePositive', value=self.FP_score)
        with cm4 :
            st.metric(label='FalseNegative', value=self.FN_score)
        
        with cm5 :
            try :
                self.accuracy = (self.TP_score+self.TN_score)/self.all_count
            except ZeroDivisionError :
                self.accuracy = 0
            
            st.metric(label='Accuracy', value=f'{self.accuracy:.3f}')
        with cm6 :
            try :
                self.precision = self.TP_score/(self.TP_score+self.FP_score)
            except ZeroDivisionError:
                self.precision = 0

            st.metric(label='Precision', value=f"{self.precision:.3f}")
        with cm7 :
            try :
                self.recall = self.TP_score/(self.TP_score+self.FN_score)
            except ZeroDivisionError :
                self.recall = 0
            st.metric(label='Recall', value=f"{self.recall:.3f}")
        with cm8 :
            try :
                self.f1_score = (2*self.recall*self.precision)/(self.recall+self.precision)
            except ZeroDivisionError :
                self.f1_score = 0

            st.metric(label='F1-score', value=f'{self.f1_score:.3f}')


    def PTM_presetting(self) :
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        in_shape = input_details[0]['shape']
        in_index = input_details[0]['index']
        out_index = output_details[0]['index']

        return interpreter, in_shape, in_index, out_index



    def preprocessing(self, data) :
        
        return data.values / 32768 # 센서의 범주가 -32768 ~ +32768

    def show_results(self) :
        # pre-work
        correct_normal_df = self.threshold_df[(self.threshold_df['result'] == "Correct") & (self.threshold_df['label'] == 1)]['filter'].reset_index(drop=True)
        correct_abnormal_df = self.threshold_df[(self.threshold_df['result'] == "Correct") & (self.threshold_df['label'] == 0)]['filter'].reset_index(drop=True)
        wrong_df = self.threshold_df[self.threshold_df['result'] != "Correct"][['filter', 'label']].reset_index(drop=True)

        correct_normal_mean = [np.mean(correct_normal_df[x::80]) for x in range(80)]
        correct_abnormal_mean = [np.mean(correct_abnormal_df[x::80]) for x in range(80)]
        correct_normal_high = [np.max(correct_normal_df[x::80]) for x in range(80)]
        correct_abnormal_high = [np.max(correct_abnormal_df[x::80]) for x in range(80)]
        correct_normal_low = [np.min(correct_normal_df[x::80]) for x in range(80)]
        correct_abnormal_low = [np.min(correct_abnormal_df[x::80]) for x in range(80)]

        correct_normal_3q = [np.quantile(correct_normal_df[x::80], .8) for x in range(80)]
        correct_abnormal_3q = [np.quantile(correct_abnormal_df[x::80], .8) for x in range(80)]
        correct_normal_1q = [np.quantile(correct_normal_df[x::80], .2) for x in range(80)]
        correct_abnormal_1q = [np.quantile(correct_abnormal_df[x::80], .2) for x in range(80)]

        # reverse
        correct_normal_low = correct_normal_low[::-1]
        correct_abnormal_low = correct_abnormal_low[::-1]
        x = [i for i in range(80)]
        x_rev = x[::-1]
        # show filter graph
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x= x+x_rev,
            y=correct_normal_3q+correct_normal_1q,
            fill='toself',
            fillcolor='rgba(0,176,246,0.4)',
            line_color='rgba(255,255,255,0)',
            showlegend=True,
            name='Normal_Range'
            ))
        fig.add_trace(go.Scatter(
            x= x+x_rev,
            y=correct_abnormal_3q+correct_abnormal_1q,
            fill='toself',
            fillcolor='rgba(231,107,243,0.4)',
            line_color='rgba(255,255,255,0)',
            showlegend=True,
            name='Abnormal_Range'
            ))
            
        fig.add_trace(go.Scatter(
            x= x+x_rev,
            y=correct_normal_high+correct_normal_low,
            fill='toself',
            fillcolor='rgba(0,176,246,0.2)',
            line_color='rgba(255,255,255,0)',
            showlegend=True,
            name='Normal_FullRange'
            ))
        fig.add_trace(go.Scatter(
            x= x+x_rev,
            y=correct_abnormal_high+correct_abnormal_low,
            fill='toself',
            fillcolor='rgba(231,107,243,0.2)',
            line_color='rgba(255,255,255,0)',
            showlegend=True,
            name='Abnormal_FullRange'
            ))
        
        fig.add_trace(go.Scatter(
            x=x,
            y=correct_normal_mean,
            line_color='rgb(0,176,246)',
            name='Normal'
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=correct_abnormal_mean,
            line_color='rgb(231,107,243)',
            name='Abormal'
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=[self.threshold for _ in range(len(x))],
            line_color='rgb(100,176,46)',
            name='Threshold'
        ))

        for e in range(len(wrong_df) // 80) :
                
            fig.add_trace(go.Scatter(
                x=x,
                y=wrong_df['filter'][e:e+80].values,
                line_color='indianred' if wrong_df['label'][e] == 1 else 'royalblue',
                name=f"WrongData{e+1}:label={'Normal' if wrong_df['label'][e] == 1 else 'Abnormal'}"
            ))

        st.plotly_chart(fig, use_container_width=True)
        fig.write_html(f'{self.threshold_df_path.replace(".csv",".html")}')


    def save_results(self) :                 
        # check extra work
        if len(self.threshold_df['result'].unique()) > 1 :
            target_df = self.threshold_df[self.threshold_df['result']=="Wrong!"]
            min_normal = self.threshold_df[self.threshold_df['label']==1]['score'].min()
            max_abnormal = self.threshold_df[self.threshold_df['label']==0]['score'].max()
            
            normal_error = target_df[target_df['label']==1]['score'].min()
            abnormal_error = target_df[target_df['label']==0]['score'].max()

            if (normal_error > max_abnormal) and (abnormal_error < min_normal) : # threshold 조정만 하면 되는 경우
                self.error_code = 1
            else :
                self.error_code = 2

        else :
            self.error_code = 0       

        log_df = pd.DataFrame({
            'model_name':self.model_name,
            'dataset':DS_OPTIONS,
            'threshold':self.threshold,
            'date': datetime.today().strftime("%Y/%m/%d %H:%M:%S"),
            'accuracy':self.accuracy,
            'precision':self.precision,
            'recall':self.recall,
            'f1-score':self.f1_score,
            'error-code':self.error_code
                            },
            index=[0])

            # 'extra_work':"" # 추가작업 필요 여부, 0 : 현상유지, 1: threashold 조정, 2: 추가학습 필요

        if os.path.exists(self.pred_log_path) :
            log_df.to_csv(self.pred_log_path, index=False, mode='a', header=False)
        else :
            log_df.to_csv(self.pred_log_path, index=False, mode='w')
        
        st.subheader("작업 종료!")
        st.dataframe(log_df)


    def show_graph(self) :

        if play :           
            
            col1, col2, col3 = st.empty(), st.empty(), st.empty()
            wrong_list = []
            data_load_state.empty()
            
            while True :    
                
                if stop :
                    break
                
                if pause :                    
                    time.sleep(10)                    

                local_time, data, label = self.load_data()
                
                # TF lite predict
                interpreter, in_shape, in_index, out_index = self.PTM_presetting()
                pre_data = self.preprocessing(data)
                in_data = np.reshape(pre_data, in_shape)
                in_data = np.array(in_data, dtype=np.float32)
                interpreter.set_tensor(in_index, in_data)

                interpreter.invoke()

                out_prob = interpreter.get_tensor(out_index)
                filter = np.reshape(out_prob, [-1])
                print("before :", filter)
                filter = pd.DataFrame(filter)[0].map(lambda x : x**10 if x > 0.95 else ( (1/(.95-x))-1 if x < 0.05 else x )).values
                print("after :", filter)
                score = np.mean(filter[:])

                pred = 1 if score > self.threshold else 0

                data_load_state.subheader(f'\tDate : \n\t{local_time}')

                if self.stop_code == 1 :
                    break
                
                self.file_cnt += 1                  
            
                with col1.container() :    
                    chart_data1 = data.iloc[:,:3] / 32768
                    st.line_chart(chart_data1, height=250)
                with col2.container() :                   
                    chart_data2 = data.iloc[:,3:] / 32768
                    st.line_chart(chart_data2, height=250)
                

                if self.file_cnt >= len(self.DATA_FILES) :
                    self.file_cnt = 0
                    self.stop_code = 1                     
                

                # result count
                if label == 1 :
                    self.real_normal += 1
                else :
                    self.real_abnormal += 1
                
                if pred == 1 :
                    self.pred_normal += 1
                else :
                    self.pred_abnormal += 1

                # result box 1
                with m1 :
                    st.metric(label='Answer', value=f"{label}", delta="정상:1 비정상:0", delta_color='off')
                with m2 :                        
                    st.metric(label='Predict', value=f"{pred}", delta=f"score={score:.6f}")

                result = "Correct" if label == pred else "Wrong!"

                if result == "Wrong!" :
                    wrong_list.append({'local_time':local_time, 'label':label, 'score':score})
                    self.wrong_score += 1
                    if label == 1 :
                        self.FN_score += 1
                    else :
                        self.FP_score += 1
                else :
                    if label == 1 :
                        self.TP_score += 1
                    else :
                        self.TN_score += 1

                self.all_count += 1

                # result box 2
                with m3 :
                    st.metric(label='Wrong Score', value=f"{self.wrong_score}", delta='누적 틀린 개수')
                with m4 :
                    st.metric(label='All Count', value=f'{self.all_count}', delta='진행된 데이터')
                
                self.show_Score()


                ### SHOW CUMULATIVE THRESHOLD DATA

                tmp_df = pd.DataFrame({
                'index':[i for i in range(len(filter))],
                'label':label,     # 실제 labeling
                'filter':filter,    # model을 통과한 각각의 score
                'score':score,     # filter normalize
                'threshold':self.threshold, # 수동으로 설정한 임계치
                'result':result,    # 정답 여부
                'local_time':local_time,# 데이터 유입 시간 - 추후 스트리밍 때 유의미
                'file_path':self.DATA_FILES[self.file_cnt-1], # streaming data가 저장된 경로                
                },).set_index('index')

                if len(self.threshold_df) > 0 :                                        
                    self.threshold_df = self.threshold_df.append(tmp_df, ignore_index=False)
                else :
                    self.threshold_df = tmp_df

                with col3.container() :  
                    st.line_chart(data = tmp_df[['filter']], use_container_width=True)
                    
        
            self.save_results()
            self.show_results()
            self.threshold_df.to_csv(self.threshold_df_path, index=False)

            with col1.container() :
                st.empty()
            with col2.container() :
                st.empty()
            with col3.container() :  
                st.empty()
                # st.line_chart(data = self.threshold_df[['filter']], use_container_width=True)   

            st.write("Wrong List")
            st.write(wrong_list)



if __name__ == "__main__" :

    st.set_page_config(
        page_title="In2Wise LwAI 시연",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/leadbreak',
            'Report a bug': "https://github.com/leadbreak",
            'About': "### This is a test app for the new skill!"
        }
    )

    st.markdown(
        f"""
            <style>
                .reportview-container .main .block-container{{                   
                    padding-top: 0rem;
                }}
            </style>
            """,
            unsafe_allow_html=True,
            )


    st.title('MLops-mini')

    add_selectbox = st.sidebar.selectbox("Now Process", ("Predict", "Pred-Log", "Training(not-yet)", "Collect(not-yet)", "Test"))

    if add_selectbox == "Pred-Log" :
        pred_log = pd.read_csv(os.path.join(os.path.expanduser('~'), "Documents\in2wise\st_lwaid\stage02\pred_log", "pred_log.csv"))
        st.dataframe(pred_log)
    elif add_selectbox == "Predict" :
            
        c1, c2, c3, c4, c5 = st.columns([2,1,1,1,1]) 
        with c1 :
            data_load_state = st.subheader('Press Start Button...')
        with c2 :
            m1 = st.empty()
        with c3 :
            m2 = st.empty()
        with c4 :
            m3 = st.empty()
        with c5 :
            m4 = st.empty()
        

        ### sidebar

        # model upload
        side_expander_01 = st.sidebar.expander("Select Predict Model & Threshold")

        with side_expander_01 :
            DS_OPTIONS = st.selectbox(
                'Select Dataset',
                ('small', 'normal')
            )      

            PTM_OPTIONS = st.selectbox(
                'Select the Pretrained Model',
                ('default', 'hpt_01', 'hpt_02', 'stupid_for_test')
            )
            
            PTM_UPLOAD = st.file_uploader('Upload your Model', type=['tflite'])
            if PTM_UPLOAD : # 자체 파일을 업로드할 경우
                # 아직 streamlit 상에서 파일 경로를 읽어오는 기능이 없으므로
                # 학습시킨 모델을 지정한 폴더(여기선 .../st_lwai/models/)의 하위 경로에 놓아야 함
                PTM_OPTIONS = os.path.join(os.path.expanduser('~'), "Documents\in2wise\st_lwaid\models", PTM_UPLOAD.name)
                #
            th_value = 0.5
            th_value = st.slider(                                            
                                    label = 'Threshold slider',
                                    min_value = 0.0, 
                                    max_value = 1.0,
                                    value=th_value,
                                )
        # button to play

        dummy1, s1, s2, s3, dummy2 = st.sidebar.columns(5)
        with s1 :
            play = st.button("▶️")
        with s2 :
            pause = st.button("⏸️") 
        with s3 :
            stop = st.button("⏹️")
        
        side_title_01 = st.sidebar.container()

        with side_title_01 :
            st1 = st.empty()
        
        s4, s5 = st.sidebar.columns(2)
        s6, s7 = st.sidebar.columns(2)
        s8, s9 = st.sidebar.columns(2)
        s10, s11 = st.sidebar.columns(2)

        # confusion matrix
        with s4 :
            cm1 = st.empty()
        with s5 :
            cm2 = st.empty()
        with s6 :
            cm3 = st.empty()
        with s7 :
            cm4 = st.empty()    

        # accuracy, precision, recall, f1-score
        with s8 :
            cm5 = st.empty()
        with s9 :
            cm6 = st.empty()
        with s10 :
            cm7 = st.empty()
        with s11 :
            cm8 = st.empty()
        

        start = st_lwai(ds_option=DS_OPTIONS, th_value=th_value, PTM_OPTIONS=PTM_OPTIONS)      
        
        start.show_graph()

        try :
            pred_log = pd.read_csv(os.path.join(os.path.expanduser('~'), "Documents\in2wise\st_lwaid\pred_log", "pred_log.csv"))
            st.subheader("Total Pred-Log Results")               
            st.dataframe(pred_log)
        except : # 아직 파일이 없을 경우
            pass

    elif add_selectbox == "Test" :
        pass