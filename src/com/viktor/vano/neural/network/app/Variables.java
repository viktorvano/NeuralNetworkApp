package com.viktor.vano.neural.network.app;

import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.layout.Pane;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;

public class Variables {
    public static final int stageHeight = 600;
    public static final int stageWidth = 800;
    public static final Pane pane = new Pane();

    public static Button buttonFile;

    public static Label labelTopologyFile, labelTrainingFile, labelTrainingStatusFile, labelWeightsFile;
    public static FileChooser fileChooser;

    public static ProgressBar progressBarTraining;

    public static Stage stageReference;

    public static File topologyFile, trainingFile, trainingStatusFile, weightsFile;
}
