package com.viktor.vano.neural.network.app;

import com.sun.istack.internal.NotNull;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.text.Font;
import javafx.stage.FileChooser;
import javafx.util.Duration;

import java.io.File;

import static com.viktor.vano.neural.network.app.GUI.GUI.customPrompt;
import static com.viktor.vano.neural.network.app.Variables.*;

public class AppFunctions {
    public static void initializeLayout()
    {
        buttonFile = new Button("File");
        buttonFile.setLayoutX(stageWidth*0.05);
        buttonFile.setLayoutY(stageHeight*0.05);
        buttonFile.setOnAction(event -> {
            topologyFile = fileChooser.showOpenDialog(stageReference);
            if (topologyFile != null) {
                System.out.println("File: " + topologyFile.getPath());
                String basePath = "";
                String fileNamingConvention = "";
                try{
                    String[] strings = topologyFile.getPath().split("topology_");

                    if(strings.length == 2 && strings[0].length() > 0)
                        basePath = strings[0];
                    else
                        customPrompt("File Chooser", "You have chosen an incorrect topology file: "
                                + topologyFile.getPath(), Alert.AlertType.WARNING);

                    strings = strings[1].split(".txt");
                    if(strings.length == 1 && strings[0].length() > 0)
                        fileNamingConvention = strings[0];
                    else
                        customPrompt("File Chooser", "Cannot determine the file naming convention from the topology file: "
                                + topologyFile.getPath(), Alert.AlertType.WARNING);
                }catch (Exception e)
                {
                    e.printStackTrace();
                    customPrompt("File Chooser", "Something went wrong picking a topology file: "
                            + topologyFile.getPath(), Alert.AlertType.ERROR);
                }

                if(topologyFile.getPath().length() > 50)
                    labelTopologyFile.setText("..." + topologyFile.getPath().substring(topologyFile.getPath().length()-50));
                else
                    labelTopologyFile.setText(topologyFile.getPath());

                trainingFile = new File(basePath + "training_" + fileNamingConvention + ".csv");
                if(!trainingFile.canRead())
                {
                    labelTrainingFile.setText("Training file not selected.");
                    customPrompt("File Chooser", "Training file can not be read: "
                            + trainingFile.getPath(), Alert.AlertType.WARNING);
                }else
                {
                    if(trainingFile.getPath().length() > 50)
                        labelTrainingFile.setText("..." + trainingFile.getPath().substring(trainingFile.getPath().length()-50));
                    else
                        labelTrainingFile.setText(trainingFile.getPath());
                }

                trainingStatusFile = new File(basePath + "trainingStatus_" + fileNamingConvention + ".txt");
                if(!trainingStatusFile.canRead())
                {
                    labelTrainingStatusFile.setText("Status file not selected.");
                    customPrompt("File Chooser", "Training status file can not be read: "
                            + trainingStatusFile.getPath(), Alert.AlertType.WARNING);
                }else
                {
                    if(trainingStatusFile.getPath().length() > 50)
                        labelTrainingStatusFile.setText("..." + trainingStatusFile.getPath().substring(trainingStatusFile.getPath().length()-50));
                    else
                        labelTrainingStatusFile.setText(trainingStatusFile.getPath());
                }

                weightsFile = new File(basePath + "weights_" + fileNamingConvention + ".dat");
                if(!weightsFile.canRead())
                {
                    labelWeightsFile.setText("Weights file not selected.");
                    customPrompt("File Chooser", "Training status file can not be read: "
                            + weightsFile.getPath(), Alert.AlertType.WARNING);
                }else
                {
                    if(weightsFile.getPath().length() > 50)
                        labelWeightsFile.setText("..." + weightsFile.getPath().substring(weightsFile.getPath().length()-50));
                    else
                        labelWeightsFile.setText(weightsFile.getPath());
                }
            }
        });
        pane.getChildren().add(buttonFile);

        labelTopologyFile = new Label("Please select a topology file.");
        labelTopologyFile.setFont(Font.font("Arial", 20));
        labelTopologyFile.setLayoutX(stageWidth*0.12);
        labelTopologyFile.setLayoutY(stageHeight*0.05);
        pane.getChildren().add(labelTopologyFile);

        labelTrainingFile = new Label("Training file not selected.");
        labelTrainingFile.setFont(Font.font("Arial", 20));
        labelTrainingFile.setLayoutX(stageWidth*0.12);
        labelTrainingFile.setLayoutY(stageHeight*0.10);
        pane.getChildren().add(labelTrainingFile);

        labelTrainingStatusFile = new Label("Status file not selected.");
        labelTrainingStatusFile.setFont(Font.font("Arial", 20));
        labelTrainingStatusFile.setLayoutX(stageWidth*0.12);
        labelTrainingStatusFile.setLayoutY(stageHeight*0.15);
        pane.getChildren().add(labelTrainingStatusFile);

        labelWeightsFile = new Label("Weights file not selected.");
        labelWeightsFile.setFont(Font.font("Arial", 20));
        labelWeightsFile.setLayoutX(stageWidth*0.12);
        labelWeightsFile.setLayoutY(stageHeight*0.20);
        pane.getChildren().add(labelWeightsFile);

        fileChooser = new FileChooser();
        fileChooser.setTitle("Open Topology File");
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Topology file", "topology*"));

        Timeline timeline = new Timeline(new KeyFrame(Duration.millis(20), event -> {
            /*if(fileSize != 0) {
                progressBarFlashedApp.setProgress((double)readChars / (double)fileSize);
                labelFlashProgress.setText("                        " + (readChars*100 / fileSize) + " %\n\n"
                        + readChars + " Bytes flashed of " +
                        fileSize + " Byte application");
            }
            else{
                progressBarFlashedApp.setProgress(0);
            labelFlashProgress.setText("                        0 %\n\n0 Bytes flashed of " +
                    fileSize + " Byte application");
            }*/
        }));
        timeline.setCycleCount(Timeline.INDEFINITE);
        timeline.play();
    }
}
