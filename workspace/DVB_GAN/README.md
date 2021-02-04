├── 01.getFeatures
|   |—— features               // Save extracted features
|   |—— makeLogFbanks.py       // Extracting audio acoustic features
|   └── run.sh                 // The sh script used to run the features extraction .py file.
├── 02.trainModel         
│   ├── model                  // save the trained model. 
│   └── train_cnn     
|       ├── net_component.py   // the neural networks.
|       ├── make_torchdata.py  // Prepare network input data.
|       ├── train.py           // Train the model. 
|       └── run.sh             // The sh script used to run train.py file.  
├── 03.inference 
|   ├── result                 // Visualization of test data.   
|   ├── net_component.py       // define neural networks.
|   ├── make_torchdata.py      //  Prepare network input data.
|   ├── inference.py           // Load the model for testing
|   └── run.sh                 // The sh script used to run inference.py file.
├── 04.inference 
|   ├── feature                // Save the extracted features. 
|   ├── net_component.py       // Define neural networks
|   ├── make_torchdata.py      // Prepare network input data. 
|   └── fm_predict.py          // Detection of FM audio files
├── 05.S2T_Baidu 
|   ├── speech_signal          // Save file detected as speech in step 04. 
|   └── Baidu_speech2text.py   // Speech-to-text conversion for audio files
├── data_fm                    // Example of Fm band
|—— dataset                    // Training data
|—— doc                        
|── README.md
└── requirements.txt     