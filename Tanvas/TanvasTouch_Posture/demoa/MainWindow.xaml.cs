using System;
using System.IO.Pipes;
using NumSharp;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Windows;
using System.Timers;
using System.Windows.Input;
using System.Security.AccessControl;
using System.Windows.Media;
using System.Windows.Interop;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using NumSharp.Generic;
using System.Numerics;
using Tanvas.TanvasTouch.Resources;
using Tanvas.TanvasTouch.WpfUtilities;
using System.Windows.Media.Imaging;
using System.Threading;
using Timer = System.Timers.Timer;
using System.Diagnostics;
using System.Security.Policy;

namespace trajectory_prediciton_test
{
    public partial class MainWindow : Window
    {
        TSprite mySprite;
        TanvasTouchViewTracker viewTracker;

        TView myView
        {

            get
            {

                return viewTracker.View;

            }
        }


        private bool timelock;
        private List<List<Point>> fingerPaths; // List of all finger stroke paths
        private Point firstPosition; // First touch point
        private bool eventlock = true; // Event lock for scene 2
        private Point lastPosition; // Last predicted point
        private List<Point> currentPath; // Current ongoing finger path
        private Timer timer; // Timer for sampling
        private double samplingInterval; // Sampling interval (in seconds)
        private DateTime startTime; // Start time of tracking
        private NamedPipeServerStream pipeServer; // Named pipe server for communication
        private bool isTracking; // Indicates if tracking is ongoing
        private bool isCounting; // Indicates if prediction should start
        private bool isFirstPoint = true; // Flag for the very first touch point
        private bool dataSaved = false; // Flag indicating if data has been saved
        private bool fingerMoving = false; // Flag for whether finger is actively moving
        private bool checkpoint = true;
        private bool scene_1 = false; // Scene 1: side-region triggering prediction
        private bool scene_2 = false; // Scene 2: center-region triggering prediction
        private bool pic = false; // Indicates whether a Tanvas image is currently displayed
        private bool enventlock = true;
        double elapsedTime = 0;
        double R_L = 0; // Right-to-left flag (0 = L→R, 1 = R→L)
        DateTime beforDT;

        // List to store multi-dimensional input sequence
        List<float[]> data = new List<float[]>();
        List<float[]> data1 = new List<float[]>();

        private InferenceSession _onnxSession;

        private void InitializeModel()
        {
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            options.AppendExecutionProvider_CPU(0); 
                                                    //options.AppendExecutionProvider_CPU(0); //Use CPU for inference (or DirectML if needed)
            string model_path = @"Resources/models/SAFTP.onnx";

            _onnxSession = new InferenceSession(model_path, options);//

        }
        public MainWindow()
        {
            InitializeModel();
            InitializeComponent();
            fingerPaths = new List<List<Point>>();
            currentPath = new List<Point>();
            samplingInterval = 0.01; // 0.01 s           

            isTracking = false;
            isCounting = false;
            // Initialize timer
            timer = new Timer(samplingInterval * 1000);
            timer.Elapsed += TimerElapsed;
            InitializeComponent();
            Tanvas.TanvasTouch.API.Initialize();// Initialize TanvasTouch API
            //Bind touch event handler
            TouchMove += MainWindow_TouchMove;

        }


      
        protected override void OnManipulationBoundaryFeedback(System.Windows.Input.ManipulationBoundaryFeedbackEventArgs e)
        {
            e.Handled = true;
            base.OnManipulationBoundaryFeedback(e);
        }
        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            pipeServer = new NamedPipeServerStream("TestPipe");
            pipeServer.WaitForConnection();
            Console.WriteLine("Connected.");
        }
  


        private void MainWindow_TouchDown(object sender, TouchEventArgs e)
        {

            beforDT = System.DateTime.Now;
            Point position = e.GetTouchPoint(canvas).Position;
            // Get the initial point
            timelock = false;

            firstPosition = position;
            Console.WriteLine($"Initial touch point：[{position}]");

            double y_f = firstPosition.Y;

            if (firstPosition.X < 500 || firstPosition.X > 1000)
            {
                // If the X position is outside the center range, enter scene 1 (used for prediction)
                scene_1 = true;
                enventlock = false;

                // Determine direction flag based on horizontal position
                if (firstPosition.X < 640)
                {
                    R_L = 0;
                }
                else
                {
                    R_L = 1;
                }
            }
            else
            {
                // If the X position is within the central region, enter scene 2
                scene_2 = true;
                eventlock = false;
            }

            // Initial image loading based on finger touch position and sliding direction
            if (R_L == 0 && y_f >= 81 && y_f <= 788)
            {
                // Left-to-right sliding: estimate the layer index based on the vertical (Y) coordinate
                double layerHeight = 707 / 210.0;
                int layer = (int)(210 - ((y_f - 81) / layerHeight) + 1);

                viewTracker = new TanvasTouchViewTracker(this);
                new BitmapImage(new Uri("../../sports_modified/Gymnastics/total/left_to_right/cluster_points_" + layer + "_inter_slope.png", UriKind.RelativeOrAbsolute)); // you own path
                var uri = new Uri("../../sports_modified/Gymnastics/total/left_to_right/cluster_points_" + layer + "_inter_slope.png"); // you own path ,
                var mySprite = PNGToTanvasTouch.CreateSpriteFromPNG(uri);
                myView.AddSprite(mySprite);
               
          }
            if(R_L == 1 && y_f >= 100 && y_f <= 700 )
            {

                // Right-to-left sliding: estimate the layer index based on Y position
                double layerHeight = 707 / 210.0;
                int layer = (int)(210 - ((y_f - 81) / layerHeight) + 1);
                viewTracker = new TanvasTouchViewTracker(this);
                new BitmapImage(new Uri("../../sports_modified/Gymnastics/total/right_to_left/cluster_points_" + layer + "_inter_slope.png", UriKind.RelativeOrAbsolute));
                var uri = new Uri("../../sports_modified/Gymnastics/total/right_to_left/cluster_points_" + layer + "_inter_slope.png");
                var mySprite = PNGToTanvasTouch.CreateSpriteFromPNG(uri);
                myView.AddSprite(mySprite);
              
            }
            if(y_f < 81 || y_f > 788)
            {
                viewTracker = new TanvasTouchViewTracker(this);
                new BitmapImage(new Uri("../../touch_test/blank.png", UriKind.RelativeOrAbsolute));
                var uri = new Uri("../../touch_test/blank.png");
                var mySprite = PNGToTanvasTouch.CreateSpriteFromPNG(uri);
                //Console.WriteLine(layer);
                myView.AddSprite(mySprite);
            }

            pic = true;
            
            
        }


        

        private void MainWindow_TouchMove(object sender, TouchEventArgs e)
        {



                        

            // Restart the timer
            timer.Start();
            // Get current finger position
            Point position = e.GetTouchPoint(canvas).Position;
            currentPath.Add(position);

            // Record time, x, and y
            float currentTime = (float)elapsedTime;
            float x = (float)position.X;
            float y = (float)position.Y;
            
         
           
            if (data.Count > 0)
            {
                float[] lastPoint = data[data.Count - 1];
                if (lastPoint[1] != x || lastPoint[2] != y)
                {
                    float[] pointData = new float[] { currentTime, x, y };
                    data.Add(pointData);
                }
            }
            else
            {
                float[] pointData = new float[] { currentTime, x, y };
                data.Add(pointData);
            }

            elapsedTime += samplingInterval;

            // For scene 1: trigger prediction every 9 points


            if (scene_1 && data.Count % 9 == 0)
            {
            


                int h = data.Count;
                Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> tensor = new DenseTensor<float>(new[] { data.Count, data[0].Length });


                if (data.Count == 1)
                {
                    return;
                }

                int direction_points = 1;
                int expand_ratio = 8; 

                
                for (int i = 0; i < data.Count; i++)
                {
                    for (int j = 0; j < data[i].Length; j++)
                    {
                        tensor[i, j] = data[i][j];
                    }
                }



                // Get last time step as center point
                float[] centered_point = new float[3];
                for (int i = 0; i < 3; i++)
                {
                    centered_point[i] = tensor[data.Count - 1, i];
                }
                // Center the sequence by subtracting center point
                for (int f = 0; f < data.Count; f++)
                {
                    for (int g = 0; g < 3; g++)
                    {
                        tensor[f, g] -= centered_point[g];
                    }
                }



                // Compute previous center point
                float[,] sequenceData = new float[data.Count, data[0].Length]; 

                for (int i = 0; i < data.Count; i++)
                {
                    for (int j = 0; j < data[0].Length; j++)
                    {
                        sequenceData[i, j] = data[i][j];
                    }
                }

                Matrix<float> sequence = Matrix<float>.Build.DenseOfArray(sequenceData);

                int numRows3 = sequence.RowCount;
                int numColumns3 = sequence.ColumnCount;

                Matrix<float> subsequence = sequence.SubMatrix(numRows3 - 1 - direction_points, direction_points, 0, numColumns3);
                MathNet.Numerics.LinearAlgebra.Vector<float> prev_centered_point = subsequence.ColumnSums() / direction_points;



                Console.WriteLine(centered_point[2]);
                float[] heading_vector = new float[2];
                for (int i = 0; i < 2; i++)
                    heading_vector[i] = centered_point[i + 1] - prev_centered_point[i + 1];




                float theta = (float)Math.Atan2(heading_vector[1], heading_vector[0]);



                //Create rotation matrix
                float[,] rotate_mat = new float[2, 2] {
                    { (float)Math.Cos(theta), -(float)Math.Sin(theta) },
                     { (float)Math.Sin(theta), (float)Math.Cos(theta) }
                     };


                // Apply rotation to all points
                for (int i = 0; i < rotate_mat.GetLength(0); i++)
                {
                    for (int j = 0; j < rotate_mat.GetLength(1); j++)
                    {
                        Console.Write(rotate_mat[i, j] + " ");
                    }
                    Console.WriteLine();
                }




                for (int i = 0; i < data.Count; i++)
                {
                    float x1 = tensor[i, 1];  // 
                    float y1 = tensor[i, 2];
                    float newX = x1 * rotate_mat[0, 0] + y1 * rotate_mat[1, 0];
                    float newY = x1 * rotate_mat[0, 1] + y1 * rotate_mat[1, 1];
                    tensor[i, 1] = newX;  //
                    tensor[i, 2] = newY;
                }






                // Set up target tensor for prediction

                int output_length = data.Count * expand_ratio;
                int seq_len = data.Count;



                Tensor<float> tgt = new DenseTensor<float>(new[] { 1, output_length - seq_len, 2 });
                int GetLength0 = tgt.Dimensions[1];


                Random random = new Random();
                for (int i = 0; i < GetLength0; i++)
                {
                    tgt[0, i, 0] = (float)random.NextDouble();
                    tgt[0, i, 1] = (float)random.NextDouble();

                }



              
                var input_np1 = tensor.Reshape(new[] { 1, data.Count, data[0].Length });
                var tgs_input = tgt.Reshape(new[] { 1, output_length - seq_len, 2 });

              
                var inputs = new List<NamedOnnxValue>
                   {
                  NamedOnnxValue.CreateFromTensor("input", input_np1),
                  NamedOnnxValue.CreateFromTensor("tgs_input", tgs_input )
                  };

             
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _onnxSession.Run(inputs);
                var resultsArray = results.ToArray();

                float[] shuchu = resultsArray[0].AsEnumerable<float>().ToArray();


                //int numElements = prediction.Length;
                int numPairs = shuchu.Length / 2;
                int numRows = shuchu.Length;
                int numColumns = 2;


                NDArray tensor2D = new NDArray(typeof(float), new Shape(numPairs, numColumns));


                for (int i = 0; i < numPairs; i++)
                {
                    tensor2D[i, 0] = shuchu[i * 2];
                    tensor2D[i, 1] = shuchu[i * 2 + 1];
                }


                for (int i = 1; i < numPairs; i++)
                {
                    tensor2D[i, 0] += tensor2D[i - 1, 0];
                    tensor2D[i, 1] += tensor2D[i - 1, 1];
                }

                float det = rotate_mat[0, 0] * rotate_mat[1, 1] - rotate_mat[0, 1] * rotate_mat[1, 0];
                float invDet = 1.0f / det;
                float[,] invRotateMat = new float[2, 2] {
                 { rotate_mat[1, 1] * invDet, -rotate_mat[0, 1] * invDet },
                 { -rotate_mat[1, 0] * invDet, rotate_mat[0, 0] * invDet }
                         };


                float[] result = new float[2];
                result[0] = invRotateMat[0, 0] + centered_point[1];
                result[1] = invRotateMat[0, 1] + centered_point[2];



                NDArray tensor2DArray = new NDArray(tensor2D.Data<float>(), tensor2D.Shape);
                NDArray resultMatrixArray = new NDArray(invRotateMat);

               
                //NDArray multiplied = np.multiply(tensor2DArray, resultMatrixArray);
                int tensor2DArrayRows = numPairs;
                int tensor2DArrayColumns = numColumns;

             
                int resultMatrixArrayRows = 2;
                int resultMatrixArrayColumns = 2;


           
                NDArray result3 = new NDArray(typeof(float), new Shape(tensor2DArrayRows, resultMatrixArrayColumns));
               
                for (int i = 0; i < tensor2DArrayRows; i++)
                {
                    float x2 = tensor2D[i, 0];
                    float y2 = tensor2D[i, 1];
                    float newX1 = x2 * invRotateMat[0, 0] + y2 * invRotateMat[1, 0];
                    float newY1 = x2 * invRotateMat[0, 1] + y2 * invRotateMat[1, 1];
                    result3[i, 0] = newX1;
                    result3[i, 1] = newY1;
                }







                float[] center_point = new float[] { centered_point[1], centered_point[2] };

                // Add the center point back to each predicted result
                for (int i = 0; i < tensor2DArrayRows; i++)
                {
                    result3[i, 0] += center_point[0];
                    result3[i, 1] += center_point[1];
                }

                float[] data3 = result3.ToArray<float>();
                Array.Resize(ref data3, data3.Length + 1);
                data3[data3.Length - 1] = (float)R_L;
                string dataString3 = string.Join(", ", data3);



                // Extract last and first few points from prediction
                float lastPositionX = result3[tensor2DArrayRows - 1, 0];
                float lastPositionY = result3[tensor2DArrayRows - 1, 1];
                float lastPositionX1 = result3[tensor2DArrayRows - 2, 0];
                float lastPositionY1 = result3[tensor2DArrayRows - 2, 1];

                float firstPositionX = result3[1, 0];
                float firstPositionY = result3[1, 1];
                float firstPositionX1 = result3[2, 0];
                float firstPositionY1 = result3[2, 1];



                // Set lastPosition for boundary check
                lastPosition.X = result3[tensor2DArrayRows - 1, 0];
                lastPosition.Y = result3[tensor2DArrayRows - 1, 1];

                
                if (lastPosition.X > 900 && firstPositionX < 500 || lastPosition.X < 500 && firstPositionX > 900)
                {

                    // Send trajectory data to Python
                    StringBuilder sb = new StringBuilder();

                    string dataToSend = sb.ToString();
                    byte[] dataBytes = Encoding.UTF8.GetBytes(dataString3);
                    pipeServer.Write(dataBytes, 0, dataBytes.Length);

                    Console.WriteLine("Data sent: " + dataToSend);
                    
                    Console.Out.Flush();

                    // Wait briefly to load new image
                    Thread.Sleep(400);
                    // Clear previous haptic overlays if present
                    if (pic && !(myView == null))
                    {

                        myView.RemoveAllSprites();
                        viewTracker.Dispose();

                    }

                    viewTracker = new TanvasTouchViewTracker(this);                   
                    var uri = new Uri("kmeans/test_fingerpath/gray_re/fingerpath_inter_slope.png");
                    var mySprite = PNGToTanvasTouch.CreateSpriteFromPNG(uri);
                    myView.AddSprite(mySprite);



                    isCounting = false;

                    isFirstPoint = false;
                    elapsedTime = 0;
                    data.Clear();
                    scene_1 = false;
                    timelock = true;
                    return;

                }
                else
                {
                    // Calculate slope for the last two points
                    Vector2 lastPoint = new Vector2(lastPositionX, lastPositionY);
                    Vector2 secondLastPoint = new Vector2(lastPositionX1, lastPositionY1);
                    Vector2 slopeLast = Vector2.Normalize(lastPoint - secondLastPoint);
                    float distanceLast = Vector2.Distance(lastPoint, secondLastPoint);

                    // List to store interpolated points on the right side
                    List<Vector2> extendedRight = new List<Vector2>();
                    List<Vector2> extendedLeft = new List<Vector2>();
                    Vector2[] newPrediction = new Vector2[extendedLeft.Count + tensor2DArrayRows + extendedRight.Count];

                    if ((firstPositionX1 - firstPositionX) > 0)
                    {

                        int bboxRightX = 900;
                     

                        // Extend the last point beyond the right boundary
                        while (!(lastPoint.X > bboxRightX))
                        {
                            lastPoint += slopeLast * distanceLast;
                            extendedRight.Add(new Vector2(lastPoint.X, lastPoint.Y));
                        }

                        // Calculate slope for the first two points
                        Vector2 firstPoint = new Vector2(firstPositionX, firstPositionY);
                        Vector2 secondPoint = new Vector2(firstPositionX1, firstPositionY1);
                        Vector2 slopeFirst = Vector2.Normalize(secondPoint - firstPoint);
                        float distanceFirst = Vector2.Distance(firstPoint, secondPoint);

                        // List to store interpolated points on the left side

                        int bboxLeftX = 200;
                       
                        // Extend the first point beyond the left boundary
                        while (!(firstPoint.X < bboxLeftX))
                        {
                            firstPoint -= slopeFirst * distanceFirst;
                            extendedLeft.Add(new Vector2(firstPoint.X, firstPoint.Y));
                        }

                        newPrediction = new Vector2[extendedLeft.Count + tensor2DArrayRows + extendedRight.Count];

                        extendedLeft.Reverse();
                        extendedLeft.CopyTo(newPrediction, 0);

                        for (int i = 0; i < tensor2DArrayRows; i++)
                        {
                            float x3 = (float)result3[i, 0];
                            float y3 = (float)result3[i, 1];
                            newPrediction[extendedLeft.Count + i] = new Vector2(x3, y3);
                        }

                        float[] data59 = new float[newPrediction.Length * 2];
                        for (int i = 0; i < newPrediction.Length; i++)
                        {
                            data59[i * 2] = newPrediction[i].X;
                            data59[i * 2 + 1] = newPrediction[i].Y;
                        }

                        Array.Resize(ref data59, data59.Length + 1);
                        data59[data59.Length - 1] = 0;
                        string dataString59 = string.Join(", ", data59);
                        


                        
                        StringBuilder sb1 = new StringBuilder();

                        string dataToSend1 = sb1.ToString();
                        byte[] dataBytes1 = Encoding.UTF8.GetBytes(dataString59);
                        pipeServer.Write(dataBytes1, 0, dataBytes1.Length);

                        Console.WriteLine("Data sent: " + dataToSend1);




                    }
                    else
                    {

                        int bboxRightX = 200;
                       
                       if(lastPosition.X < 1280) 
                        {
                        // Extend the last point beyond the right boundary
                        while (!(lastPoint.X < bboxRightX))
                        {
                            lastPoint += slopeLast * distanceLast;
                            extendedRight.Add(new Vector2(lastPoint.X, lastPoint.Y));
                        }
                        }
                        // Calculate slope for the first two points
                        Vector2 firstPoint = new Vector2(firstPositionX, firstPositionY);
                        Vector2 secondPoint = new Vector2(firstPositionX1, firstPositionY1);
                        Vector2 slopeFirst = Vector2.Normalize(secondPoint - firstPoint);
                        float distanceFirst = Vector2.Distance(firstPoint, secondPoint);

                        // List to store interpolated points on the left side

                        int bboxLeftX = 1200;
                     
                        // Extend the first point beyond the left boundary
                        while (!(firstPoint.X > bboxLeftX))
                        {
                            firstPoint -= slopeFirst * distanceFirst;
                            extendedLeft.Add(new Vector2(firstPoint.X, firstPoint.Y));
                        }

                        newPrediction = new Vector2[extendedLeft.Count + tensor2DArrayRows + extendedRight.Count];

                        extendedLeft.Reverse();
                        extendedLeft.CopyTo(newPrediction, 0);

                        for (int i = 0; i < tensor2DArrayRows; i++)
                        {
                            float x3 = (float)result3[i, 0];
                            float y3 = (float)result3[i, 1];
                            newPrediction[extendedRight.Count + i] = new Vector2(x3, y3);
                        }
                    extendedRight.CopyTo(newPrediction, extendedLeft.Count + tensor2DArrayRows);

                        float[] data9 = new float[newPrediction.Length * 2];
                        for (int i = 0; i < newPrediction.Length; i++)
                        {
                            data9[i * 2] = newPrediction[i].X;
                            data9[i * 2 + 1] = newPrediction[i].Y;
                        }
                        Array.Resize(ref data9, data9.Length + 1);
                        data9[data9.Length - 1] = 1.0f;
                        string dataString39 = string.Join(", ", data9);
                        



                        // sent data to python 
                        StringBuilder sb3 = new StringBuilder();

                        string dataToSend = sb3.ToString();
                        byte[] dataBytes = Encoding.UTF8.GetBytes(dataString39);
                        pipeServer.Write(dataBytes, 0, dataBytes.Length);

                        Console.WriteLine("Data sent: " + dataToSend);
                        //Reset state

                        elapsedTime = 0;
                        data.Clear();
                        extendedLeft.Clear();
                        extendedRight.Clear();

                        Console.Out.Flush();

                    }


                    Thread.Sleep(400);
                  
                    if (pic && !(myView==null))
                    {

                        myView.RemoveAllSprites();
                        viewTracker.Dispose();

                    }

                    viewTracker = new TanvasTouchViewTracker(this);
                    var uri = new Uri("/kmeans/test_fingerpath/gray_re/fingerpath_inter_slope.png");
                    var mySprite = PNGToTanvasTouch.CreateSpriteFromPNG(uri);
                    myView.AddSprite(mySprite);
                    isCounting = false;

                    isFirstPoint = false;
                    elapsedTime = 0;
                    data.Clear();
                    scene_1 = false;
                    timelock = true;
                    return;
                   



                }
            }




            if (scene_2 && data.Count % 2 == 0)
            {
                double deltaX1 = firstPosition.X - position.X;
                double deltaY1 = firstPosition.Y - position.Y;

               


                int h = data.Count;
                Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> tensor = new DenseTensor<float>(new[] { data.Count, data[0].Length });


                if (data.Count == 1)
                {
                    return;
                }

                int direction_points = 1;
                int expand_ratio = 8; 


                for (int i = 0; i < data.Count; i++)
                {
                    for (int j = 0; j < data[i].Length; j++)
                    {
                        tensor[i, j] = data[i][j];
                    }
                }


                // Get last time step for centering
                float[] centered_point = new float[3];
                for (int i = 0; i < 3; i++)
                {
                    centered_point[i] = tensor[data.Count - 1, i];
                }
                Console.WriteLine(string.Join(",", centered_point));
                // Center the tensor by subtracting the last point
                for (int f = 0; f < data.Count; f++)
                {
                    for (int g = 0; g < 3; g++)
                    {
                        tensor[f, g] -= centered_point[g];
                    }
                }


                // Convert data list to 2D float array
                float[,] sequenceData = new float[data.Count, data[0].Length]; 

                for (int i = 0; i < data.Count; i++)
                {
                    for (int j = 0; j < data[0].Length; j++)
                    {
                        sequenceData[i, j] = data[i][j];
                    }
                }

                Matrix<float> sequence = Matrix<float>.Build.DenseOfArray(sequenceData);

                int numRows3 = sequence.RowCount;
                int numColumns3 = sequence.ColumnCount;

                Matrix<float> subsequence = sequence.SubMatrix(numRows3 - 1 - direction_points, direction_points, 0, numColumns3);
                MathNet.Numerics.LinearAlgebra.Vector<float> prev_centered_point = subsequence.ColumnSums() / direction_points;


                string dataString10 = string.Join(", ", prev_centered_point);
         

                Console.WriteLine(centered_point[2]);
                float[] heading_vector = new float[2];
                for (int i = 0; i < 2; i++)
                    heading_vector[i] = centered_point[i + 1] - prev_centered_point[i + 1];


                string dataString12 = string.Join(", ", heading_vector);
       


                float theta = (float)Math.Atan2(heading_vector[1], heading_vector[0]);
                float[,] rotate_mat = new float[2, 2] {
                    { (float)Math.Cos(theta), -(float)Math.Sin(theta) },
                     { (float)Math.Sin(theta), (float)Math.Cos(theta) }
                     };


                // Apply rotation to each (x, y) point
                for (int i = 0; i < rotate_mat.GetLength(0); i++)
                {
                    for (int j = 0; j < rotate_mat.GetLength(1); j++)
                    {
                        Console.Write(rotate_mat[i, j] + " ");
                    }
                    Console.WriteLine();
                }


                for (int i = 0; i < data.Count; i++)
                {
                    float x1 = tensor[i, 1];  
                    float y1 = tensor[i, 2];
                    float newX = x1 * rotate_mat[0, 0] + y1 * rotate_mat[1, 0];
                    float newY = x1 * rotate_mat[0, 1] + y1 * rotate_mat[1, 1];
                    tensor[i, 1] = newX;  
                    tensor[i, 2] = newY;
                }





                int output_length = data.Count * expand_ratio;
                int seq_len = data.Count;



                Tensor<float> tgt = new DenseTensor<float>(new[] { 1, output_length - seq_len, 2 });
                int GetLength0 = tgt.Dimensions[1];


                Random random = new Random();
                float fixedValue = 1.0f;
                for (int i = 0; i < GetLength0; i++)
                {

                     tgt[0, i, 0] = fixedValue;
                     tgt[0, i, 1] = fixedValue;
                }





               

                var input_np1 = tensor.Reshape(new[] { 1, data.Count, data[0].Length });
                var tgs_input = tgt.Reshape(new[] { 1, output_length - seq_len, 2 });

              
                var inputs = new List<NamedOnnxValue>
                   {
                  NamedOnnxValue.CreateFromTensor("input", input_np1),
                  NamedOnnxValue.CreateFromTensor("tgs_input", tgs_input )
                  };



                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _onnxSession.Run(inputs);
                var resultsArray = results.ToArray();

                float[] shuchu = resultsArray[0].AsEnumerable<float>().ToArray();





                int numPairs = shuchu.Length / 2;
                int numRows = shuchu.Length;
                int numColumns = 2;

 
                NDArray tensor2D = new NDArray(typeof(float), new Shape(numPairs, numColumns));

                for (int i = 0; i < numPairs; i++)
                {
                    tensor2D[i, 0] = shuchu[i * 2];
                    tensor2D[i, 1] = shuchu[i * 2 + 1];
                }


                for (int i = 1; i < numPairs; i++)
                {
                    tensor2D[i, 0] += tensor2D[i - 1, 0];
                    tensor2D[i, 1] += tensor2D[i - 1, 1];
                }




                float det = rotate_mat[0, 0] * rotate_mat[1, 1] - rotate_mat[0, 1] * rotate_mat[1, 0];
                float invDet = 1.0f / det;
                float[,] invRotateMat = new float[2, 2] {
                 { rotate_mat[1, 1] * invDet, -rotate_mat[0, 1] * invDet },
                 { -rotate_mat[1, 0] * invDet, rotate_mat[0, 0] * invDet }
                         };

                float[] result = new float[2];
                result[0] = invRotateMat[0, 0] + centered_point[1];
                result[1] = invRotateMat[0, 1] + centered_point[2];




                for (int i = 0; i < invRotateMat.GetLength(0); i++)
                {
                    for (int j = 0; j < invRotateMat.GetLength(1); j++)
                    {
                        Console.Write(invRotateMat[i, j] + " ");
                    }
                    Console.WriteLine();
                }



                NDArray tensor2DArray = new NDArray(tensor2D.Data<float>(), tensor2D.Shape);
                NDArray resultMatrixArray = new NDArray(invRotateMat);


                int tensor2DArrayRows = numPairs;
                int tensor2DArrayColumns = numColumns;


                int resultMatrixArrayRows = 2;
                int resultMatrixArrayColumns = 2;



                NDArray result3 = new NDArray(typeof(float), new Shape(tensor2DArrayRows, resultMatrixArrayColumns));

                for (int i = 0; i < tensor2DArrayRows; i++)
                {
                    float x2 = tensor2D[i, 0];
                    float y2 = tensor2D[i, 1];
                    float newX1 = x2 * invRotateMat[0, 0] + y2 * invRotateMat[1, 0];
                    float newY1 = x2 * invRotateMat[0, 1] + y2 * invRotateMat[1, 1];
                    result3[i, 0] = newX1;
                    result3[i, 1] = newY1;
                }







                float[] center_point = new float[] { centered_point[1], centered_point[2] };


                for (int i = 0; i < tensor2DArrayRows; i++)
                {
                    result3[i, 0] += center_point[0];
                    result3[i, 1] += center_point[1];
                }

                float[] data3 = result3.ToArray<float>();
                string dataString3 = string.Join(", ", data3);

                isCounting = false;

                isFirstPoint = false;

                // Define lastPosition 
                lastPosition.X = (float)result3[tensor2DArrayRows - 1, 0];
                lastPosition.Y = (float)result3[tensor2DArrayRows - 1, 1];
                //  prediction array
                int[,] prediction1 = new int[1, 2];

               
                float lastPositionX = result3[tensor2DArrayRows - 1, 0];
                float lastPositionY = result3[tensor2DArrayRows - 1, 1];
                float lastPositionX1 = result3[tensor2DArrayRows - 2, 0];
                float lastPositionY1 = result3[tensor2DArrayRows - 2, 1];

                float firstPositionX = result3[1, 0];
                float firstPositionY = result3[1, 1];
                float firstPositionX1 = result3[2, 0];
                float firstPositionY1 = result3[2, 1];


                Vector2[] newPrediction = null;
                if (lastPositionX > 900 && firstPositionX < 500 || lastPositionX < 500 && firstPositionX > 900)
                {
                    elapsedTime = 0;
                    data.Clear();
                    scene_2 = false;


                    StringBuilder sb = new StringBuilder();

                    string dataToSend = sb.ToString();
                    byte[] dataBytes = Encoding.UTF8.GetBytes(dataString3);
                    pipeServer.Write(dataBytes, 0, dataBytes.Length);

                    Console.WriteLine("Data sent: " + dataToSend);


                    elapsedTime = 0;
                    data.Clear();

                    Console.Out.Flush();

                    Thread.Sleep(400);

                    if (pic && !(myView == null))
                    {

                        myView.RemoveAllSprites();
                        viewTracker.Dispose();

                    }

                    viewTracker = new TanvasTouchViewTracker(this);
                    var uri = new Uri("kmeans/test_fingerpath/gray_re/fingerpath_inter_slope.png");
                    var mySprite = PNGToTanvasTouch.CreateSpriteFromPNG(uri);
                    myView.AddSprite(mySprite);
                    return;
                    

                }




                else
                {

                    // Left to right
                    if ((firstPositionX1 - firstPositionX) > 0)
                    {

                        // Calculate slope for the last two points
                        Vector2 lastPoint = new Vector2((float)lastPositionX, (float)lastPositionY);
                        Vector2 secondLastPoint = new Vector2(lastPositionX1, lastPositionY1);
                        Vector2 slopeLast = Vector2.Normalize(lastPoint - secondLastPoint);
                        float distanceLast = Vector2.Distance(lastPoint, secondLastPoint);

                        // List to store interpolated points on the right side
                        List<Vector2> extendedRight = new List<Vector2>();
                        int bboxRightX = 1200;
                        int bboxRightY = 69;

                        // Extend the last point beyond the right boundary
                        while (!(lastPoint.X > bboxRightX))
                        {
                            lastPoint += slopeLast * distanceLast;
                            extendedRight.Add(new Vector2(lastPoint.X, lastPoint.Y));
                        }

                        // Calculate slope for the first two points
                        Vector2 firstPoint = new Vector2(firstPositionX, firstPositionY);
                        Vector2 secondPoint = new Vector2(firstPositionX1, firstPositionY1);
                        Vector2 slopeFirst = Vector2.Normalize(secondPoint - firstPoint);
                        float distanceFirst = Vector2.Distance(firstPoint, secondPoint);

                        // List to store interpolated points on the left side

                        List<Vector2> extendedLeft = new List<Vector2>();
                        int bboxLeftX = 200;
                        int bboxLeftY = 746;


                        // Extend the first point beyond the left boundary
                        while (!(firstPoint.X < bboxLeftX))
                        {
                            firstPoint -= slopeFirst * distanceFirst;
                            extendedLeft.Add(new Vector2(firstPoint.X, firstPoint.Y));
                        }

                        newPrediction = new Vector2[extendedLeft.Count + tensor2DArrayRows + extendedRight.Count];
                        extendedLeft.Reverse();
                        extendedLeft.CopyTo(newPrediction, 0);

                        for (int i = 0; i < tensor2DArrayRows; i++)
                        {
                            float x3 = (float)result3[i, 0];
                            float y3 = (float)result3[i, 1];
                            newPrediction[extendedLeft.Count + i] = new Vector2(x3, y3);
                        }

                        extendedRight.CopyTo(newPrediction, extendedLeft.Count + tensor2DArrayRows);

                        float[] data9 = new float[newPrediction.Length * 2];
                        for (int i = 0; i < newPrediction.Length; i++)
                        {
                            data9[i * 2] = newPrediction[i].X;
                            data9[i * 2 + 1] = newPrediction[i].Y;
                        }


                        Array.Resize(ref data9, data9.Length + 1);
                        data9[data9.Length - 1] = 0;
                        string dataString39 = string.Join(", ", data9);





                        StringBuilder sb = new StringBuilder();

                        string dataToSend = sb.ToString();
                        byte[] dataBytes = Encoding.UTF8.GetBytes(dataString39);
                        pipeServer.Write(dataBytes, 0, dataBytes.Length);

                        Console.WriteLine("Data sent: " + dataToSend);

                        Thread.Sleep(400);
 
                        if (pic && !(myView == null))
                        {

                            myView.RemoveAllSprites();
                            viewTracker.Dispose();

                        }

                        viewTracker = new TanvasTouchViewTracker(this);
                        var uri = new Uri("kmeans/test_fingerpath/gray_re/fingerpath_inter_slope.png");
                        var mySprite = PNGToTanvasTouch.CreateSpriteFromPNG(uri);
                        myView.AddSprite(mySprite);

                        isFirstPoint = false;
                        elapsedTime = 0;
                        data.Clear();
                        scene_2 = false;
                        timelock = true;

                    }
                    //Right to left
                    else
                    {
                        // Calculate slope for the last two points
                        Vector2 lastPoint1 = new Vector2((float)lastPositionX, (float)lastPositionY);
                        Vector2 secondLastPoint = new Vector2(lastPositionX1, lastPositionY1);
                        Vector2 slopeLast = Vector2.Normalize(lastPoint1 - secondLastPoint);
                        float distanceLast = Vector2.Distance(lastPoint1, secondLastPoint);

                        // List to store interpolated points on the right side

                        List<Vector2> extendedRight = new List<Vector2>();
                        int bboxRightX = 400;
                        int bboxRightY = 69;

                        // Extend the last point beyond the right boundary
                        while (!(lastPoint1.X < bboxRightX))
                        {
                            lastPoint1 += slopeLast * distanceLast;
                            extendedRight.Add(new Vector2(lastPoint1.X, lastPoint1.Y));
                        }
                        // Calculate slope for the first two points
                        Vector2 firstPoint1 = new Vector2(firstPositionX, firstPositionY);
                        Vector2 secondPoint = new Vector2(firstPositionX1, firstPositionY1);
                        Vector2 slopeFirst = Vector2.Normalize(firstPoint1 - secondPoint);
                        float distanceFirst = Vector2.Distance(firstPoint1, secondPoint);

                        // List to store interpolated points on the left side
                        List<Vector2> extendedLeft = new List<Vector2>();
                        int bboxLeftX = 1200;
                        int bboxLeftY = 746;
                        // Extend the first point beyond the left boundary
                        while (!(firstPoint1.X > bboxLeftX))
                        {
                            firstPoint1 += slopeFirst * distanceFirst;
                            extendedLeft.Add(new Vector2(firstPoint1.X, firstPoint1.Y));
                        }

                        newPrediction = new Vector2[extendedLeft.Count + tensor2DArrayRows + extendedRight.Count];
                        extendedLeft.Reverse();
                        extendedLeft.CopyTo(newPrediction, 0);

                        for (int i = 0; i < tensor2DArrayRows; i++)
                        {
                            float x3 = (float)result3[i, 0];
                            float y3 = (float)result3[i, 1];
                            newPrediction[extendedLeft.Count + i] = new Vector2(x3, y3);
                        }

                        extendedRight.CopyTo(newPrediction, extendedLeft.Count + tensor2DArrayRows);

                        
                        isCounting = false;

                        float[] data9 = new float[newPrediction.Length * 2];
                        for (int i = 0; i < newPrediction.Length; i++)
                        {
                            data9[i * 2] = newPrediction[i].X;
                            data9[i * 2 + 1] = newPrediction[i].Y;
                        }
                        Array.Resize(ref data9, data9.Length + 1);
                        data9[data9.Length - 1] = 1.0f;
                        string dataString39 = string.Join(", ", data9);



                        StringBuilder sb = new StringBuilder();

                        string dataToSend = sb.ToString();
                        byte[] dataBytes = Encoding.UTF8.GetBytes(dataString39);
                        pipeServer.Write(dataBytes, 0, dataBytes.Length);

                        Console.WriteLine("Data sent: " + dataToSend);

                        Thread.Sleep(400);
                         
                        if (pic && !(myView == null))
                        {

                            myView.RemoveAllSprites();
                            viewTracker.Dispose();

                        }

                        viewTracker = new TanvasTouchViewTracker(this);
                        var uri = new Uri("kmeans/test_fingerpath/gray_re/fingerpath_inter_slope.png");
                        var mySprite = PNGToTanvasTouch.CreateSpriteFromPNG(uri);
                        myView.AddSprite(mySprite);

                        isFirstPoint = false;
                        elapsedTime = 0;
                        data.Clear();
                        scene_2 = false;
                        timelock = true;



                    }




                    elapsedTime = 0;
                    data.Clear();

                    Console.Out.Flush();


               
                }
            
            }
            if(timelock )
            { 
            DateTime afterDT = System.DateTime.Now;
            TimeSpan ts = afterDT.Subtract(beforDT);
            Console.WriteLine("Elapsed time: {0} ms.", ts.TotalMilliseconds);
            timelock = false;
            }

        }

        private void MainWindow_TouchUp(object sender, TouchEventArgs e)
        {
         
            elapsedTime = 0;
            data.Clear();

            isFirstPoint = true;
            Console.WriteLine("Node 2 - eventlock: " + eventlock);
            scene_1 = false;
            scene_2 = false;
            eventlock = true;


            // Clear any sprites from the view
            if (myView != null)
            {
                myView.RemoveAllSprites();
                viewTracker.Dispose();
            }
                                  
            pic = false;

        

        }


        private void TimerElapsed(object sender, ElapsedEventArgs e)
        {
            // Check time has passed while finger is moving
            TimeSpan elapsedTime = DateTime.Now - startTime;

            if (elapsedTime.TotalSeconds >= 1 && fingerMoving)
            {
                
                timer.Stop();

            }
        }

    }
}


