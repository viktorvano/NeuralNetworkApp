package com.viktor.vano.neural.network.app;

import com.viktor.vano.neural.network.app.FFNN.NeuralNetParameters;
import com.viktor.vano.neural.network.app.FFNN.NeuralNetwork;
import javafx.scene.control.*;
import javafx.scene.layout.Pane;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;
import java.util.ArrayList;

public class Variables {
    public static int stageHeight = 600;
    public static int stageWidth = 800;
    public static final int bottomOffset = 80;
    public static final Pane pane = new Pane();
    public static Button buttonFile, buttonTrain, buttonRandomRun;
    public static Label labelTopologyFile, labelTrainingFile, labelTrainingStatusFile, labelWeightsFile;
    public static FileChooser fileChooser;
    public static ProgressBar progressBarTraining;
    public static Stage stageReference;
    public static File topologyFile, trainingFile, trainingStatusFile, weightsFile;
    public static boolean filesOK = true;
    public static NeuralNetParameters neuralNetParameters;
    public static NeuralNetwork neuralNetwork;
    public static ArrayList<ArrayList<Button>> buttonNeurons;
    public static ArrayList<Slider> sliderInputs;
    public static ArrayList<TextField> textFieldInputs;

    public static boolean isBusy = false;
}
