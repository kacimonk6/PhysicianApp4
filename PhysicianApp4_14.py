import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import boto3
from io import StringIO, BytesIO
from colorama import init, Fore, Style
import streamlit as st
import re
import bcrypt
import sqlite3
import secrets
import time
from streamlit_option_menu import option_menu
from scipy.signal import butter,filtfilt
from time import time_ns, sleep
import sys
import csv
import os
import re




st.set_page_config(page_title="LetSense", layout="wide")


init(autoreset=True)  # Initialize colorama to auto-reset colors after each print statement
# Connect to SQLite database (or create it)
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT, email TEXT)''')
conn.commit()

#Add security question/answer to the database
try:
    c.execute('ALTER TABLE users ADD COLUMN security_question TEXT')
    c.execute('ALTER TABLE users ADD COLUMN security_answer TEXT')
except sqlite3.OperationalError:
    pass

# Function to hash passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Function to check passwords
def check_password(hashed_password, user_password):
    return bcrypt.checkpw(user_password.encode(), hashed_password.encode())

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'register' not in st.session_state:
    st.session_state['register'] = False
if 'forgot_password' not in st.session_state:
    st.session_state['forgot_password'] = False
if 'reset_user' not in st.session_state:
    st.session_state['reset_user'] = None

#Function to trigger rereun
def rerun():
    st.session_state['rerun'] = not st.session_state.get('rereun', False)




# Letrep Logo
col1, col2, col3, col4, col5 = st.columns(5)
st.markdown("")


# Only display the logo if the user is not logged in
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    with col3:
        st.image('./WebsiteLogo.png', width=370)


# Only display the login page title if the user is not logged in
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("Login Page")


# Registration form
if 'register' not in st.session_state:
    st.session_state['register'] = False

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    if st.button('Register', type='primary'):
            st.session_state['register'] = True

if st.session_state['register']:
    new_username = st.text_input('New Username')
    new_password = st.text_input('New Password', type='password')
    security_question = st.text_input('Security Question')
    security_answer = st.text_input('Security Answer')
    if st.button('Submit Registration'):
        hashed_password = hash_password(new_password)
        hashed_answer = hash_password(security_answer)
        c.execute('INSERT INTO users (username, password, security_question, security_answer) VALUES (?, ?, ?, ?)', (new_username, hashed_password, security_question, hashed_answer))
        conn.commit()
        st.success('User registered successfully')
        st.session_state['register'] = False
        st.session_state['logged_in'] = True   # Set the user as logged in
        st.session_state['username'] = new_username  # Save the new username in session state
        st.rerun()

# Login form
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login', type = 'primary'):
        c.execute('SELECT password, email FROM users WHERE username = ?', (username,))
        result = c.fetchone()
        if result and check_password(result[0], password):
            st.success(f'Welcome {username}')
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.rerun()
        else:
            st.error('Username/password is incorrect')
    
# Forgot Password form
if not st.session_state['logged_in'] and st.button('Forgot Password', type= 'primary'):
    st.session_state['forgot_password'] = True

if st.session_state['forgot_password']:
    username = st.text_input('Enter your username')
    if st.button('Submit Username', type = 'primary'):
        c.execute('SELECT security_question FROM users WHERE username = ?', (username,))
        result = c.fetchone()
        if result:
            st.session_state['reset_user'] = username
            st.session_state['security_question'] = result[0]
        else:
            st.error('Username not found')

# Answer security question form
if st.session_state['reset_user']:
    security_answer = st.text_input(f"Answer: {st.session_state['security_question']}")
    if st.button('Submit Answer', type = 'primary'):
        c.execute('SELECT security_answer FROM users WHERE username = ?', (st.session_state['reset_user'],))
        result = c.fetchone()
        if result and check_password(result[0], security_answer):
            st.session_state['reset_token'] = secrets.token_urlsafe(16)
            st.session_state['reset_expiry'] = time.time() + 3600  # Token valid for 1 hour
            st.success('Security question answered correctly. Please reset your password.')
        else:
            st.error('Incorrect answer')

# Password reset form
if st.session_state.get('reset_token') and time.time() < st.session_state.get('reset_expiry', 0):
    new_password = st.text_input('Enter new password', type='password')
    if st.button('Reset Password', type = 'primary'):
        hashed_password = hash_password(new_password)
        c.execute('UPDATE users SET password = ? WHERE username = ?', (hashed_password, st.session_state['reset_user']))
        conn.commit()
        st.success('Password reset successfully')
        st.session_state['reset_user'] = None
        st.session_state['reset_token'] = None
        st.cache_resource.clear() 
        

#Display pages only when the user is logged in            
if st.session_state['logged_in']:
    st.sidebar.image('./WebsiteLogo.png', width=270)
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.success("You have been logged out. Please log in again.")
        st.rerun()

    selected = option_menu(None, ["Home", "User Data", "Pelvic Tilt"], 
            icons=["house", "clipboard-data", "person"], 
            menu_icon="cast", 
            default_index=1,
            orientation = 'horizontal'
            
    )
    
    
# Display the selected page content
    if selected == "Home":
        st.title("Home Page")
        st.write("")
        st.write(f"Welcome {st.session_state['username']}!")
        st.write("")         
        st.subheader("Welcome to the LetSense Physician App Home Page.")

    elif selected == "User Data":
        st.title("User Data Page")
        st.write("")
        st.write("")
            
        st.subheader("Select each IMU file from the bucket")

        # Select files from the S3 bucket 
        aws_credentials = st.secrets["aws"]
        access_key_id = aws_credentials["access_key_id"]
        secret_access_key = aws_credentials["secret_access_key"]
        region = aws_credentials["region"]
        bucket_name = aws_credentials["bucket_name"]

        # S3 client
        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region
        )
        s3_client = session.client('s3')

        # List files in S3 bucket (this code should only run when "User Data" is selected)
        files = s3_client.list_objects_v2(Bucket=bucket_name).get('Contents', [])
        file_names = [file['Key'] for file in files]

        names = set()
        for file_name in file_names:
            if file_name.startswith("IMU00"):
                name_part = file_name.split('_')[1]
                names.add(name_part)
        selected_name = st.selectbox("Select patient name", sorted(names))
        imu00_filtered = [file for file in file_names if file.startswith(f"IMU00_{selected_name}_")]

        # Filter iles based on IMU00 or IMU01
        #imu00_filter = [file for file in file_names if file.startswith("IMU00")]
        #imu01_filter= [file for file in file_names if file.startswith("IMU01")]

        # Let the user select IMU files
        imu00_files= st.selectbox("Select IMU00 File", imu00_filtered)
        #imu01_files = st.selectbox("Select IMU01 CSV file", imu01_filter)
        imu01_filtered = [file for file in file_names if file.startswith(f"IMU01_{selected_name}_")]
        
        
        imu01_files = st.selectbox("Select IMU01 File", imu01_filtered)

        st.write("<span style='color:red;'>⚠️ Please make sure the IMU files have the same name, date, and session number</span>", unsafe_allow_html=True)
            
        if imu00_files and imu01_files:
                st.write(f"Selected IMU Files: {imu00_files}, {imu01_files}")
            #common_name = imu00_files.split('IMU00_')[1]

            #imu01_files_filtered = [file for file in imu01_filter if file.endswith(common_name)]

            #imu01_files = st.selectbox("select IMU01 File", imu01_files_filtered)
            #if imu01_files:
                

                #st.write(f"Selected IMU Files: {imu00_files}, {imu01_files}")

                # Read the CSV files from S3 directly
                imu00_obj = s3_client.get_object(Bucket=bucket_name, Key=imu00_files)
                imu01_obj = s3_client.get_object(Bucket=bucket_name, Key=imu01_files)

                df00 = pd.read_csv(imu00_obj['Body'])
                df01 = pd.read_csv(imu01_obj['Body'])
            
            
                def rotation_matrix(alpha, beta, gamma):
                    R = np.zeros((3, 3))
                    
                    R[0, 0] = np.cos(gamma) * np.cos(beta)
                    R[0, 1] = np.cos(gamma) * np.sin(beta) * np.sin(alpha) + np.sin(gamma) * np.cos(alpha)
                    R[0, 2] = np.sin(gamma) * np.sin(alpha) - np.cos(gamma) * np.sin(beta) * np.cos(alpha)

                    R[1, 0] = -np.sin(gamma) * np.cos(beta)
                    R[1, 1] = np.cos(alpha) * np.cos(gamma) - np.sin(gamma) * np.sin(beta) * np.sin(alpha)
                    R[1, 2] = np.sin(gamma) * np.sin(beta) * np.cos(alpha) + np.cos(gamma) * np.sin(alpha)

                    R[2, 0] = np.sin(beta)
                    R[2, 1] = -np.cos(beta) * np.sin(alpha)
                    R[2, 2] = np.cos(beta) * np.cos(alpha)
                    
                    return R
                
                # Sample Rate and Initialization
                SampleRate = 10
                nRows = min(df00.shape[0], df01.shape[0])

                # Initialize Cardan Lumbar Angles
                Cardan_lumbar = np.zeros((nRows, 3))
                
                # Loop through the rows to compute the lumbar angles
                for k in range(nRows):

                    #Pelvis (Base)
                    alpha00 = df00.iloc[k, 5]
                    beta00 = -df00.iloc[k, 6]
                    gamma00 = df00.iloc[k, 7]

                    #Sternum (Trunk)
                    alpha01 = -df01.iloc[k, 5]
                    beta01 = df01.iloc[k, 6]
                    gamma01 = df01.iloc[k, 7]
                    

                    # Compute rotation matrices for trunk and base
                    R_trunk = rotation_matrix(alpha01, beta01, gamma01)
                    R_base = rotation_matrix(alpha00, beta00, gamma00)

                    # Compute relative rotation matrix (trunk -> base)
                    #R_trunk_base = np.dot(R_trunk, R_base.T)
                    R_trunk_base = np.matmul(R_trunk, R_base.T)
                    
                    # Cardan Angles
                    alpha = np.arctan2(-R_trunk_base[2, 1], R_trunk_base[2, 2])
                    beta = np.arcsin(R_trunk_base[2, 0])
                    gamma = np.arctan(-R_trunk_base[1, 0] / R_trunk_base[0, 0])

                    Cardan_lumbar[k] = [alpha, beta, gamma]

                    Cardan_lumbar_deg = Cardan_lumbar* 180 / np.pi
                    time = np.linspace(0, (nRows - 1) * 1/SampleRate, nRows)
                    #output_df = pd.DataFrame(np.hstack([time[:, None], Cardan_lumbar_deg]), columns=['Time', 'FlexionExtension', 'LateralBending', 'Rotation'])

                    # Save the output
                    #output_df.to_csv(f'LumbarAngles_{fileName}', sep='\t', index=False)


            ############################## NEW CODE#############################################
                N_rows=np.size(Cardan_lumbar, axis = 0)
                Angle0=0 #initialize the number of points in the data between -20 and 0 deg (for example)
                Angle1=0 #initialize the number of points in the data between 0 and 20 deg 
                Angle2=0 #initialize the number of points in the data between 20 and 40 deg 
                Angle3=0 #initialize the number of points in the data between 40 and 60 deg 
                Angle4=0 #initialize the number of points in the data outside of -20 and +60

                for i in range(0,N_rows):
                    if -20<=Cardan_lumbar_deg[i,0] and Cardan_lumbar_deg[i,0]<0:
                        Angle0=Angle0+1
                    elif 0<=Cardan_lumbar_deg[i,0] and Cardan_lumbar_deg[i,0]<20:
                        Angle1=Angle1+1
                    elif 20<=Cardan_lumbar_deg[i,0] and Cardan_lumbar_deg[i,0]<40:
                        Angle2=Angle2+1
                    elif 40<=Cardan_lumbar_deg[i,0] and Cardan_lumbar_deg[i,0]<60:
                        Angle3=Angle3+1
                    else:
                        Angle4=Angle4+1

                Angle0_percent=100*Angle0/N_rows   
                Angle1_percent=100*Angle1/N_rows
                Angle2_percent=100*Angle2/N_rows
                Angle3_percent=100*Angle3/N_rows
                Angle4_percent=100*Angle4/N_rows   

                #bar graph
                x = np.array(["-20-0 deg", "0-20 deg", "20-40 deg", "40-60 deg", "Outside"])
                y = np.array([Angle0_percent,Angle1_percent,Angle2_percent,Angle3_percent,Angle4_percent])
                time = nRows/SampleRate

                plt.xlabel('Angle Categories')
                
                
                time_duration_text = "Time duration: " + str(time) + " sec"
                plt.figtext(0.9, 0.02, time_duration_text, fontsize=7, ha="right", va="bottom")
                
                fig, ax = plt.subplots(figsize=(6,4))
            
                fig.subplots_adjust(bottom=0.1, top=1.2)
                # Set the background color for the figure (entire plot)
                fig.patch.set_facecolor("#d7e8e8")  # Change this to any color you prefer

                # Set the background color for the axes (plot area)
                ax.set_facecolor("#d7e8e8")
                
                    # Plot bars with color
                bars = ax.bar(x, y, color=['#42bbc7'] * len(x))

                # Add exact % labels on top of each bar
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 1, # adjust height offset as needed
                        f'{y[i]:.1f}%', # display the actual AngleX_percent value
                        ha='center',
                        va='bottom',
                        fontsize=10
                        )

                
                
                plt.ylabel('% Duration')
                plt.title("Summary Data for 1 Session")
                bar_colors = ['#42bbc7']  # Example: Different colors for each bar
                plt.bar(x, y, color=bar_colors)
                
                plt.ylabel('% Duration')
                plt.title("Summary Data for 1 Session")
                plt.bar(x,y)
                
                image_bytes= BytesIO()
                plt.savefig(image_bytes, format='png')
                image_bytes.seek(0) 
                
                st.pyplot(plt)
                
                
                st.download_button(
                    label="Download Bar Graph",
                    data=image_bytes,
                    file_name="bar_graph.png",  ###change to be name_date_session_bar
                    mime="image/png"
                )
        
        elif selected == "Pelvic Tilt":
                st.title("Pelvic Tilt")
                st.write("")
                st.write("")

                from colorama import init, Fore, Style
                init(autoreset=True)  # Initialize colorama to auto-reset colors after each print statement

                    # Initialize constants and globals
                MICROSECONDS_IN_A_SECOND = 1000000
                NANOSECONDS_IN_A_SECOND = 1000000000
                PORT = None  # Autodetect the port, modify if necessary
                TIMEOUT = 0.7
                DESIRED_SAMPLES_PER_SECOND = 100

                    # Initialize control variables
                recording = False  # State variable to track recording status
                files_open = False  # Tracks if the files are open
                csvfile_0 = None
                writer_0 = None
                iteration = 0
                exit_flag = False 

                    # CONSTANTS / GLOBALS
                MICROSECONDS_IN_A_SECOND = 1000000  # 1 million, 6 zeros
                us = MICROSECONDS_IN_A_SECOND
                NANOSECONDS_IN_A_SECOND = 1000000000  # 1 billion, 9 zeros
                ns = NANOSECONDS_IN_A_SECOND

                PORT = None  # we will try to autodetect the port
                    # change/comment out if needed
                    # PORT = '/dev/ttyACM0'  # pi
                    # PORT = "COM26"  # windows

                LOGICAL_IDS = [0]

                    # change if needed. raise if there are data transfer problems
                TIMEOUT = 0.7

                    # change if needed. 0 should be fine. using 1 as an example.
                START_STREAM_DELAY_IN_SECONDS = 1

                    # change if needed
                TIME_TO_WAIT_BEFORE_GIVING_UP = 5

                    # change if needed
                DESIRED_SAMPLES_PER_SECOND = 60
                hz = DESIRED_SAMPLES_PER_SECOND


                if st.button("Initialize Sensors"):
                    st.write("Getting com class")
                    senCom = USB_ExampleClassStreamlit.UsbCom(PORT, timeout=TIMEOUT)
                    st.write("Getting tss class (if you hang here, double check your COM port)")
                    senTSS = None
                try:
                        senTSS = ThreeSpaceSensor(senCom, streamingBufferLen=1000)
                except:
                        #st.write(Fore.YELLOW + "Resetting dongle/sensor, please wait about 10 seconds.")
                        #st.write(Fore.YELLOW + "If the program hangs, power cycle your dongle / sensors and restart the program.")
                        senCom.write(":226\n".encode('latin'),None)  # resets dongle/sensor
                        success = False
                        while not success:
                            try:
                                senTSS = ThreeSpaceSensor(senCom, streamingBufferLen=1000)
                                success = True
                                for id in LOGICAL_IDS:
                                    sleep(0)
                                    senTSS.stopStreaming(logicalID=id)
                            except:
                                sleep(1)
                        sleep(0)
                senTSS.comClass.sensor.reset_input_buffer()

                for id in LOGICAL_IDS:
                        senTSS.setAxisDirections(1, logicalID=id)

                def tareSensors(): 
                        st.write("Taring sensors")
                        for id in LOGICAL_IDS:
                            senTSS.tareWithCurrentOrientation(logicalID=id)

                st.write("Setting streaming slots")
                for id in LOGICAL_IDS:
                        senTSS.setStreamingSlots(Streamable.READ_TARED_ORIENTATION_AS_EULER, 
                                                Streamable.READ_TARED_ORIENTATION_AS_QUAT, logicalID=id)
                st.write("Setting response header bitfield")
                    # headerConfig=0x1+0x2+0x4+0x8+0x10+0x20+0x40
                    # senTSS.setResponseHeaderBitfield(headerConfig=0x1+0x2+0x4+0x8+0x10+0x20+0x40)
                    #sesenTSS.setResponseHeaderBitfield(headerConfig=0x1+0x2+0x4)

                st.write("Setting streaming timing")
                interval = 0
                if hz != 0:
                        interval = round(float(us)/hz)
                for id in LOGICAL_IDS:
                        senTSS.setStreamingTiming(
                            interval=interval,
                            duration=STREAM_CONTINUOUSLY,  # 0xFFFFFFFF
                            delay=START_STREAM_DELAY_IN_SECONDS*us,
                            logicalID=id
                        )

                    # additional setup
                packets = {}
                for id in LOGICAL_IDS:
                        packets[id] = []


                st.write("Starting stream...")
                for id in LOGICAL_IDS:
                        senTSS.startStreaming(logicalID=id)

                eepy = 0.9/(float(hz)*len(LOGICAL_IDS))
                eepy = 0

                sleep(3)
                for i in range(10):
                        for id in LOGICAL_IDS:
                            senTSS.getOldestStreamingPacket(logicalID=id)
                        sleep(eepy)
                        sleep(3)
                st.write("Streaming Started")
                        
                start_time = time_ns()
                    # Function definitions


                def find_next_iteration_number():
                        # Pattern to match the files, assuming they are named like 'IMU00_Duration_Collection_01.csv'
                        pattern = re.compile(r'IMU\d{2}_Duration_Collection_(\d{2})\.csv')
                        highest_iteration = 0

                        # List all files in the current directory
                        for filename in os.listdir('.'):
                            match = pattern.match(filename)
                            if match:
                                # Extract iteration number from filename and convert to int
                                iteration_number = int(match.group(1))
                                if iteration_number > highest_iteration:
                                    highest_iteration = iteration_number

                        # Return the next iteration number
                        return highest_iteration + 1


                def start_recording(patient_name, date, session):
                        global recording, csvfile_0, writer_0, iteration
                        # Only start recording if not already recording
                        if not recording:
                            winsound.Beep(400, 50) # Beep at 400 Hz for 50 milliseconds
                            winsound.Beep(800, 70) # Beep at 800 Hz for 70 milliseconds

                            recording = True
                            # Find next iteration number
                            iteration = find_next_iteration_number()
                            # Open new CSV files for writing
                            st.session_state.csvfile_0 = open(f'IMU00_Duration_Collection_{iteration:02d}.csv', 'w', newline='')
                            # Create new CSV writers
                            st.session_state.writer_0 = csv.writer(csvfile_0)
                            st.write("Recording started: Session {iteration}")

                def stop_recording():
                        global recording, csvfile_0, csvfile_1, writer_0, exit_flag
                        # Only stop recording if currently recording
                        if recording:
                            winsound.Beep(700, 50) # Beep at 700 Hz for 50 milliseconds
                            winsound.Beep(300, 70) # Beep at 300 Hz for 70 milliseconds
                            recording = False
                            # Close the files if they are open
                            if st.session_state.csvfile_0 is not None:
                                st.session_state.csvfile_0.close()
                                st.session_state.csvfile_0 = None  # Reset file object to None
                            # Reset writers to None
                            st.session_state.writer_0 = None
                            st.write("Recording stopped")
                            exit_flag = True


                def record_data(senTSS):
                        for id in LOGICAL_IDS:
                            packet = senTSS.getOldestStreamingPacket(logicalID=id)
                            if packet is not None and writer_0 is not None:
                                if id == 0:
                                    st.session_state.writer_0.writerow(packet)
                    
                with col1:
                        st.markdown(
                            """
                            <style>
                            .custom-label {
                                font-size: 24px;
                                font-weight; bold;
                                color: #333;
                            }
                            .custom-input input {
                                font-size:20px !important;
                                height: 60px !important;
                                width: 500px !important;
                                padding: 10px !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                        )
                        st.markdown('<p class="custom-label"> Please enter the name of the patient:</p>', unsafe_allow_html=True)
                        patient_name = st.text_input("", key="input", placeholder= "e.g. John Doe", label_visibility="collapsed")
                        st.markdown('<p class="custom-label"> Please enter the date:</p>', unsafe_allow_html=True)
                        date = st.text_input(" ", key=input, placeholder="(e.g., 11_7_24)", label_visibility= "collapsed")
                        st.markdown('<p class="custom-label"> Please enter the session number:</p>', unsafe_allow_html=True)
                        session = st.text_input("", placeholder="e.g. 01", label_visibility="collapsed")

                        st.markdown(
                            """
                                <script>
                                var elements = window.parent.document.querySelectorAll('.stTextInput input');
                                elements[elements.length-1].classList.add('custom-input-box');
                                </script>
                                """,
                                unsafe_allow_html=True
                            )
        
                        st.markdown(
                                """
                                <style>
                                .stAlert {
                                    font-size:32px !important;
                                    padding 30px !important
                                }
                                </style>
                                """,
                                unsafe_allow_html = True,
                            )
                    # Main loop
                try:
                    st.write("Press start button to begin recording and press stop to terminate the recording")
                        
                    if st.button("Tare Sensors"):
                        tareSensors(senTSS, LOGICAL_IDS)
                        
                    if st.button("Start Recording"):
                            start_recording()
                    if st.button("Stop Recording"):
                            stop_recording()
                            
                            winsound.Beep(200, 200)  # Beep at 200 Hz for 200 milliseconds
                            last_time_check = time.time()     
                    while not exit_flag:
                            if recording:
                                record_data()
                                time_clock_time = time.time()
                                if time_clock_time - last_time_check >= 10: #stops the data collection automatically after 6 hours 
                                    stop_recording()  
                            sleep(eepy)  # Adjust as needed to manage CPU usage
                except KeyboardInterrupt:
                        st.write("Program terminated by user.")
                finally:
                        winsound.Beep(900, 50) # Beep at 900 Hz for 50 milliseconds
                        winsound.Beep(500, 70) # Beep at 500 Hz for 70 milliseconds
                        # Cleanup
                        if st.session_state.csvfile_0 is not None:
                            st.session_state.csvfile_0.close()
                        st.write("Cleaned up files.")
                        st.write("Stopping stream")
                        for id in LOGICAL_IDS:
                            senTSS.stopStreaming(logicalID=id)
                        st.write("Closed Stream! Goodbye!")
                        winsound.Beep(600, 50) # Beep at 600 Hz for 50 milliseconds
                        winsound.Beep(400, 70) # Beep at 200 Hz for 70 milliseconds


