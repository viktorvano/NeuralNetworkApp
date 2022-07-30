package com.viktor.vano.neural.network.app;

import com.sun.istack.internal.NotNull;
import com.viktor.vano.neural.network.app.FFNN.NeuralNetParameters;
import com.viktor.vano.neural.network.app.FFNN.NeuralNetwork;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.text.Font;
import javafx.stage.FileChooser;
import javafx.util.Duration;

import java.io.File;
import java.util.ArrayList;

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
                filesOK = true;
                System.out.println("File: " + topologyFile.getPath());
                String basePath = "";
                String fileNamingConvention = "";
                try{
                    String[] strings = topologyFile.getPath().split("topology_");

                    if(strings.length == 2 && strings[0].length() > 0)
                        basePath = strings[0];
                    else
                    {
                        filesOK = false;
                        customPrompt("File Chooser", "You have chosen an incorrect topology file: "
                                + topologyFile.getPath(), Alert.AlertType.WARNING);
                    }

                    strings = strings[1].split(".txt");
                    if(strings.length == 1 && strings[0].length() > 0)
                        fileNamingConvention = strings[0];
                    else
                    {
                        filesOK = false;
                        customPrompt("File Chooser", "Cannot determine the file naming convention from the topology file: "
                                + topologyFile.getPath(), Alert.AlertType.WARNING);
                    }
                }catch (Exception e)
                {
                    e.printStackTrace();
                    filesOK = false;
                    customPrompt("File Chooser", "Something went wrong picking a topology file: "
                            + topologyFile.getPath(), Alert.AlertType.WARNING);
                }

                if(topologyFile.getPath().length() > 50)
                    labelTopologyFile.setText("..." + topologyFile.getPath().substring(topologyFile.getPath().length()-50));
                else
                    labelTopologyFile.setText(topologyFile.getPath());

                trainingFile = new File(basePath + "training_" + fileNamingConvention + ".csv");
                if(!trainingFile.canRead())
                {
                    filesOK = false;
                    customPrompt("File Chooser", "Training file can not be read: "
                            + trainingFile.getPath(), Alert.AlertType.WARNING);
                }

                if(trainingFile.getPath().length() > 50)
                    labelTrainingFile.setText("..." + trainingFile.getPath().substring(trainingFile.getPath().length()-50));
                else
                    labelTrainingFile.setText(trainingFile.getPath());


                trainingStatusFile = new File(basePath + "trainingStatus_" + fileNamingConvention + ".txt");
                if(!trainingStatusFile.canRead())
                {
                    customPrompt("File Chooser", "Training status file does not exist: "
                            + trainingStatusFile.getPath(), Alert.AlertType.INFORMATION);
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
                    customPrompt("File Chooser", "Weights file can not be read: "
                            + weightsFile.getPath() + "\nNew file will be created.", Alert.AlertType.INFORMATION);
                }

                if(weightsFile.getPath().length() > 50)
                    labelWeightsFile.setText("..." + weightsFile.getPath().substring(weightsFile.getPath().length()-50));
                else
                    labelWeightsFile.setText(weightsFile.getPath());

                if(neuralNetParameters == null && filesOK)
                {
                    neuralNetParameters = new NeuralNetParameters(topologyFile.getPath(), trainingFile.getPath(),
                                                                    weightsFile.getPath(), trainingStatusFile.getPath(),
                            0.1f,0.5f, 0.001f, 5000, 1000000);
                    neuralNetwork = new NeuralNetwork(neuralNetParameters);

                    buttonNeurons = new ArrayList<>();
                    for (int i = 0; i < neuralNetParameters.topology.size(); i++)
                    {
                        buttonNeurons.add(new ArrayList<>());
                        for(int l = 0; l < neuralNetParameters.topology.get(i); l++)
                        {
                            buttonNeurons.get(i).add(new Button(i + " - " + l));
                            buttonNeurons.get(i).get(l).setLayoutX(0.1*stageWidth +
                                    (0.9*stageWidth / ((float)neuralNetParameters.topology.size())) * i);
                            buttonNeurons.get(i).get(l).setLayoutY(0.25*stageHeight +
                                    (0.75*stageHeight / ((float)neuralNetParameters.topology.get(i))) * l +
                                    ((stageHeight / ((float)neuralNetParameters.topology.get(i))) / 2) - 50);
                            pane.getChildren().add(buttonNeurons.get(i).get(l));
                        }
                    }
                }else if(neuralNetParameters != null && !filesOK)
                {
                    if(neuralNetwork != null && neuralNetwork.isNetTraining())
                        neuralNetwork.stopTraining();

                    for (ArrayList<Button> setOfNeuronButtons : buttonNeurons)
                    {
                        for (Button buttonNeuron : setOfNeuronButtons)
                        {
                            pane.getChildren().remove(buttonNeuron);
                        }
                        setOfNeuronButtons.clear();
                    }
                    buttonNeurons.clear();

                    buttonNeurons = null;
                    neuralNetwork = null;
                    neuralNetParameters = null;
                }else if(neuralNetParameters != null)
                {
                    if(neuralNetwork != null && neuralNetwork.isNetTraining())
                        neuralNetwork.stopTraining();

                    for (ArrayList<Button> setOfNeuronButtons : buttonNeurons)
                    {
                        for (Button buttonNeuron : setOfNeuronButtons)
                        {
                            pane.getChildren().remove(buttonNeuron);
                        }
                        setOfNeuronButtons.clear();
                    }
                    buttonNeurons.clear();

                    neuralNetParameters = new NeuralNetParameters(topologyFile.getPath(), trainingFile.getPath(),
                            weightsFile.getPath(), trainingStatusFile.getPath(),
                            0.1f,0.5f, 0.001f, 5000, 1000000);
                    neuralNetwork = new NeuralNetwork(neuralNetParameters);

                    buttonNeurons = new ArrayList<>();
                    for (int i = 0; i < neuralNetParameters.topology.size(); i++)
                    {
                        buttonNeurons.add(new ArrayList<>());
                        for(int l = 0; l < neuralNetParameters.topology.get(i); l++)
                        {
                            buttonNeurons.get(i).add(new Button(i + " - " + l));
                            buttonNeurons.get(i).get(l).setLayoutX(0.1*stageWidth +
                                    (0.9*stageWidth / ((float)neuralNetParameters.topology.size())) * i);
                            buttonNeurons.get(i).get(l).setLayoutY(0.25*stageHeight +
                                    (0.75*stageHeight / ((float)neuralNetParameters.topology.get(i))) * l +
                                    ((stageHeight / ((float)neuralNetParameters.topology.get(i))) / 2) - 50);
                            pane.getChildren().add(buttonNeurons.get(i).get(l));
                        }
                    }
                }
            }
        });
        pane.getChildren().add(buttonFile);

        labelTopologyFile = new Label("Please select a topology file.");
        labelTopologyFile.setFont(Font.font("Arial", 16));
        labelTopologyFile.setLayoutX(stageWidth*0.12);
        labelTopologyFile.setLayoutY(stageHeight*0.05);
        pane.getChildren().add(labelTopologyFile);

        labelTrainingFile = new Label("Training file not selected.");
        labelTrainingFile.setFont(Font.font("Arial", 16));
        labelTrainingFile.setLayoutX(stageWidth*0.12);
        labelTrainingFile.setLayoutY(stageHeight*0.08);
        pane.getChildren().add(labelTrainingFile);

        labelTrainingStatusFile = new Label("Status file not selected.");
        labelTrainingStatusFile.setFont(Font.font("Arial", 16));
        labelTrainingStatusFile.setLayoutX(stageWidth*0.12);
        labelTrainingStatusFile.setLayoutY(stageHeight*0.11);
        pane.getChildren().add(labelTrainingStatusFile);

        labelWeightsFile = new Label("Weights file not selected.");
        labelWeightsFile.setFont(Font.font("Arial", 16));
        labelWeightsFile.setLayoutX(stageWidth*0.12);
        labelWeightsFile.setLayoutY(stageHeight*0.14);
        pane.getChildren().add(labelWeightsFile);

        fileChooser = new FileChooser();
        fileChooser.setTitle("Open Topology File");
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Topology file", "topology*"));

        buttonTrain = new Button("Train");
        buttonTrain.setLayoutX(stageWidth*0.85);
        buttonTrain.setLayoutY(stageHeight*0.05);
        buttonTrain.setDisable(true);
        buttonTrain.setOnAction(event -> {
            if(neuralNetwork != null)
            {
                neuralNetwork.trainNeuralNetwork();
            }
        });
        pane.getChildren().add(buttonTrain);

        Timeline timelineRefresh = new Timeline(new KeyFrame(Duration.millis(250), event -> {
            buttonTrain.setDisable(neuralNetwork == null ||  neuralNetwork.isNetTraining());
            buttonFile.setDisable(neuralNetwork != null && neuralNetwork.isNetTraining());
        }));
        timelineRefresh.setCycleCount(Timeline.INDEFINITE);
        timelineRefresh.play();
    }
}
