package com.viktor.vano.neural.network.app;

import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.text.Font;
import javafx.stage.FileChooser;
import javafx.util.Duration;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

import static com.viktor.vano.neural.network.app.Variables.*;
import static com.viktor.vano.neural.network.app.GUI.GUI.*;

public class AppFunctions {
    public static void initializeLayout()
    {
        buttonFile = new Button("File");
        buttonFile.setLayoutX(stageWidth*0.05);
        buttonFile.setLayoutY(stageHeight*0.20);
        buttonFile.setOnAction(event -> {
            file = fileChooser.showOpenDialog(stageReference);
            if (file != null) {
                System.out.println("File: " + file.getPath());
                /*try {
                    fileSize = Files.size(Paths.get(file.getPath()));
                    labelFileSize.setText("File size: " + fileSize + " Bytes");
                    progressBarMemory.setProgress(fileSize/22528.0);
                    labelFileSizeBar.setText(Math.round((fileSize/22528.0)*10000.0)/100.0 + " % of 22 528 Byte Memory");
                    binaryContent = new byte[(int) fileSize];
                    InputStream inputStream = new FileInputStream(file);
                    inputStream.read(binaryContent);
                    readChars = 0;
                    if(fileSize > 22528)
                        customPrompt("File Error",
                                "File is too big: " + fileSize + " Bytes\nThat is " +
                                        (Math.round((fileSize/22528.0)*10000.0)/100.0)
                                        + " % of 22 528 Byte Memory", Alert.AlertType.ERROR);
                } catch (IOException e) {
                    e.printStackTrace();
                }*/
                if(file.getPath().length() > 50)
                    labelFile.setText("..." + file.getPath().substring(file.getPath().length()-50));
                else
                    labelFile.setText(file.getPath());
                //buttonFlash.setDisable(fileSize == 0 || btnConnect.getText().equals("Connect"));
            }
        });
        pane.getChildren().add(buttonFile);

        labelFile = new Label("Please select a file.");
        labelFile.setFont(Font.font("Arial", 20));
        labelFile.setLayoutX(stageWidth*0.12);
        labelFile.setLayoutY(stageHeight*0.205);
        pane.getChildren().add(labelFile);

        fileChooser = new FileChooser();
        fileChooser.setTitle("Open Binary File");
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
