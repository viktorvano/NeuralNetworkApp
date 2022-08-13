package com.viktor.vano.neural.network.app;

import com.viktor.vano.neural.network.app.FFNN.NeuralNetParameters;
import com.viktor.vano.neural.network.app.FFNN.NeuralNetwork;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.scene.control.*;
import javafx.scene.text.Font;
import javafx.stage.FileChooser;
import javafx.util.Duration;

import java.io.File;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Locale;

import static com.viktor.vano.neural.network.app.FFNN.GeneralFunctions.showVectorValues;
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
                            buttonNeurons.get(i).add(new Button(formatFloatToString4(0f)));
                            buttonNeurons.get(i).get(l).setStyle(colorStyle(0f));
                            buttonNeurons.get(i).get(l).setPrefSize(70, 40);
                            buttonNeurons.get(i).get(l).setLayoutX(0.2*stageWidth +
                                    (0.8*stageWidth / ((float)neuralNetParameters.topology.size())) * i);
                            buttonNeurons.get(i).get(l).setLayoutY(0.26*stageHeight +
                                    (0.75*stageHeight / ((float)neuralNetParameters.topology.get(i))) * l +
                                    ((stageHeight / ((float)neuralNetParameters.topology.get(i))) / 2) - bottomOffset);
                            pane.getChildren().add(buttonNeurons.get(i).get(l));
                        }
                    }

                    sliderInputs = new ArrayList<>();
                    textFieldInputs = new ArrayList<>();
                    for(int l = 0; l < neuralNetParameters.topology.get(0); l++)
                    {
                        sliderInputs.add(new Slider(-1.0, 1.0, 0));
                        //sliderInputs.get(l).setPrefSize(120, 40);
                        sliderInputs.get(l).setLayoutX(0.05*stageWidth);
                        sliderInputs.get(l).setLayoutY(0.24*stageHeight +
                                (0.75*stageHeight / ((float)neuralNetParameters.topology.get(0))) * l +
                                ((stageHeight / ((float)neuralNetParameters.topology.get(0))) / 2) - bottomOffset);
                        pane.getChildren().add(sliderInputs.get(l));

                        textFieldInputs.add(new TextField());
                        textFieldInputs.get(l).setPrefSize(120, 40);
                        textFieldInputs.get(l).setLayoutX(0.05*stageWidth);
                        textFieldInputs.get(l).setLayoutY(0.28*stageHeight +
                                (0.75*stageHeight / ((float)neuralNetParameters.topology.get(0))) * l +
                                ((stageHeight / ((float)neuralNetParameters.topology.get(0))) / 2) - bottomOffset);
                        pane.getChildren().add(textFieldInputs.get(l));
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
                            buttonNeurons.get(i).add(new Button(formatFloatToString4(0f)));
                            buttonNeurons.get(i).get(l).setStyle(colorStyle(0f));
                            buttonNeurons.get(i).get(l).setPrefSize(70, 40);
                            buttonNeurons.get(i).get(l).setLayoutX(0.2*stageWidth +
                                    (0.8*stageWidth / ((float)neuralNetParameters.topology.size())) * i);
                            buttonNeurons.get(i).get(l).setLayoutY(0.26*stageHeight +
                                    (0.75*stageHeight / ((float)neuralNetParameters.topology.get(i))) * l +
                                    ((stageHeight / ((float)neuralNetParameters.topology.get(i))) / 2) - bottomOffset);
                            pane.getChildren().add(buttonNeurons.get(i).get(l));
                        }
                    }


                    for (Slider slider : sliderInputs)
                    {
                        pane.getChildren().remove(slider);
                    }
                    sliderInputs.clear();

                    for (TextField textField : textFieldInputs)
                    {
                        pane.getChildren().remove(textField);
                    }
                    textFieldInputs.clear();


                    sliderInputs = new ArrayList<>();
                    textFieldInputs = new ArrayList<>();
                    for(int l = 0; l < neuralNetParameters.topology.get(0); l++)
                    {
                        sliderInputs.add(new Slider(-1.0, 1.0, 0));
                        //sliderInputs.get(l).setPrefSize(120, 40);
                        sliderInputs.get(l).setLayoutX(0.05*stageWidth);
                        sliderInputs.get(l).setLayoutY(0.24*stageHeight +
                                (0.75*stageHeight / ((float)neuralNetParameters.topology.get(0))) * l +
                                ((stageHeight / ((float)neuralNetParameters.topology.get(0))) / 2) - bottomOffset);
                        pane.getChildren().add(sliderInputs.get(l));

                        textFieldInputs.add(new TextField());
                        textFieldInputs.get(l).setPrefSize(120, 40);
                        textFieldInputs.get(l).setLayoutX(0.05*stageWidth);
                        textFieldInputs.get(l).setLayoutY(0.28*stageHeight +
                                (0.75*stageHeight / ((float)neuralNetParameters.topology.get(0))) * l +
                                ((stageHeight / ((float)neuralNetParameters.topology.get(0))) / 2) - bottomOffset);
                        pane.getChildren().add(textFieldInputs.get(l));
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

        buttonRandomRun = new Button("Random Run");
        buttonRandomRun.setLayoutX(stageWidth*0.85);
        buttonRandomRun.setLayoutY(stageHeight*0.10);
        buttonRandomRun.setDisable(true);
        buttonRandomRun.setOnAction(event -> {
            if(neuralNetwork != null)
            {
                runCycleOfNN();
            }
        });
        pane.getChildren().add(buttonRandomRun);

        Timeline timelineRefresh = new Timeline(new KeyFrame(Duration.millis(250), event -> {
            buttonTrain.setDisable(neuralNetwork == null ||  neuralNetwork.isNetTraining() || isBusy);
            buttonRandomRun.setDisable(neuralNetwork == null ||  neuralNetwork.isNetTraining() || isBusy);
            buttonFile.setDisable(neuralNetwork != null && (neuralNetwork.isNetTraining() || isBusy));

            if(stageReference.getWidth() != stageWidth || stageReference.getHeight() != stageHeight)
            {
                stageWidth = (int)stageReference.getWidth();
                stageHeight = (int)stageReference.getHeight();
                updateLayoutPositions();
                System.out.println("Updated layout from timeline.");
            }

            if(!isBusy && buttonNeurons != null)
            {
                for (int i = 0; i < buttonNeurons.size(); i++)
                {
                    for(int l = 0; l < buttonNeurons.get(i).size(); l++)
                    {
                        buttonNeurons.get(i).get(l).setText(formatFloatToString4(neuralNetwork.getNeuronOutput(i,l)));
                        buttonNeurons.get(i).get(l).setStyle(colorStyle(neuralNetwork.getNeuronOutput(i,l)));
                    }
                }
            }
        }));
        timelineRefresh.setCycleCount(Timeline.INDEFINITE);
        timelineRefresh.play();
    }

    public static String colorStyle(float value)
    {
        float color = value;
        if(value > 1.0f)
            value = 1.0f;
        else if(value < -1.0f)
            value = -1.0f;

        value *= 255.0f;
        boolean positive = true;
        if(value >= 0.0f)
        {
            positive=true;
        }else
        {
            positive = false;
            value = -value;
        }
        String hexString = Integer.toHexString((int)value);

        StringBuilder stringBuilder = new StringBuilder("-fx-background-color: #");
        String colorString = null;
        if (positive)
        {
            if (hexString.length()>1)
                stringBuilder.append("00" + hexString + "00;");
            else
                stringBuilder.append("000" + hexString + "00;");
        }else
        {
            if (hexString.length()>1)
                stringBuilder.append("0000" + hexString + ";");
            else
                stringBuilder.append("00000" + hexString + ";");
        }
        if(color > 0.6)
            stringBuilder.append(" -fx-text-fill: black;");
        else
            stringBuilder.append(" -fx-text-fill: white;");
        colorString = stringBuilder.toString();

        return colorString;
    }

    public static String formatFloatToString4(float number)
    {
        DecimalFormatSymbols formatSymbols = new DecimalFormatSymbols(Locale.getDefault());
        formatSymbols.setDecimalSeparator('.');
        return new DecimalFormat("##########.####", formatSymbols).format(number);
    }

    public static void updateLayoutPositions()
    {
        buttonFile.setLayoutX(stageWidth*0.05);
        buttonFile.setLayoutY(stageHeight*0.05);

        labelTopologyFile.setLayoutX(stageWidth*0.12);
        labelTopologyFile.setLayoutY(stageHeight*0.05);

        labelTrainingFile.setLayoutX(stageWidth*0.12);
        labelTrainingFile.setLayoutY(stageHeight*0.08);

        labelTrainingStatusFile.setLayoutX(stageWidth*0.12);
        labelTrainingStatusFile.setLayoutY(stageHeight*0.11);

        labelWeightsFile.setLayoutX(stageWidth*0.12);
        labelWeightsFile.setLayoutY(stageHeight*0.14);

        buttonTrain.setLayoutX(stageWidth*0.85);
        buttonTrain.setLayoutY(stageHeight*0.05);

        buttonRandomRun.setLayoutX(stageWidth*0.85);
        buttonRandomRun.setLayoutY(stageHeight*0.10);

        if(neuralNetParameters != null && filesOK)
        {
            for (int i = 0; i < neuralNetParameters.topology.size(); i++)
            {
                for(int l = 0; l < neuralNetParameters.topology.get(i); l++)
                {
                    buttonNeurons.get(i).get(l).setLayoutX(0.2*stageWidth +
                            (0.8*stageWidth / ((float)neuralNetParameters.topology.size())) * i);
                    buttonNeurons.get(i).get(l).setLayoutY(0.26*stageHeight +
                            (0.75*stageHeight / ((float)neuralNetParameters.topology.get(i))) * l +
                            ((stageHeight / ((float)neuralNetParameters.topology.get(i))) / 2) - bottomOffset);
                }
            }

            for(int l = 0; l < neuralNetParameters.topology.get(0); l++)
            {
                sliderInputs.get(l).setLayoutX(0.05*stageWidth);
                sliderInputs.get(l).setLayoutY(0.24*stageHeight +
                        (0.75*stageHeight / ((float)neuralNetParameters.topology.get(0))) * l +
                        ((stageHeight / ((float)neuralNetParameters.topology.get(0))) / 2) - bottomOffset);

                textFieldInputs.get(l).setLayoutX(0.05*stageWidth);
                textFieldInputs.get(l).setLayoutY(0.28*stageHeight +
                        (0.75*stageHeight / ((float)neuralNetParameters.topology.get(0))) * l +
                        ((stageHeight / ((float)neuralNetParameters.topology.get(0))) / 2) - bottomOffset);
            }
        }
    }

    public static void runCycleOfNN()
    {
        isBusy = true;
        neuralNetwork.neuralNetParameters.input.clear();
        for(int i = 0; i < neuralNetwork.neuralNetParameters.topology.get(0); i++)
        {
            neuralNetwork.neuralNetParameters.input.add((float)Math.random());
        }
        showVectorValues("Inputs:", neuralNetwork.neuralNetParameters.input);
        neuralNetwork.feedForward(neuralNetwork.neuralNetParameters.input);

        assert(neuralNetwork.neuralNetParameters.input.size() ==
                neuralNetwork.neuralNetParameters.topology.get(0));

        // Collect the net's actual results:
        neuralNetwork.getResults(neuralNetwork.neuralNetParameters.result);
        showVectorValues("Outputs: ", neuralNetwork.neuralNetParameters.result);

        isBusy = false;
    }
}
