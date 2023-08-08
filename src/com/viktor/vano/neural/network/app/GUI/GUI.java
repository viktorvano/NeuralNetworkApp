package com.viktor.vano.neural.network.app.GUI;

import com.sun.istack.internal.NotNull;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.scene.image.Image;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import javafx.util.Duration;

import java.util.Optional;

import static com.viktor.vano.neural.network.app.AppFunctions.*;
import static com.viktor.vano.neural.network.app.Variables.*;

public class GUI extends Application {


    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) {
        stage.setTitle("Neural Network App - no UI v" + versionNumber);
        Scene scene = new Scene(pane, stageWidth, stageHeight);
        stage.setMinHeight(stageHeight);
        stage.setMinWidth(stageWidth);
        stage.setResizable(true);
        stage.setScene(scene);
        stage.getIcons().add(new Image("/com/viktor/vano/neural/network/app/resources/icon.jpg"));
        stage.show();

        stage.widthProperty().addListener((observable, oldValue, newValue) -> {
            stageWidth = newValue.intValue();
            updateLayoutPositions();
        });

        stage.heightProperty().addListener((observable, oldValue, newValue) -> {
            stageHeight = newValue.intValue();
            updateLayoutPositions();
        });

        stageReference = stage;

        initializeLayout();
    }

    @Override
    public void stop() throws Exception {
        super.stop();
        imagination.stop();
        if(neuralNetwork != null && neuralNetwork.isNetTraining())
            neuralNetwork.stopTraining();
    }

    public static void customPrompt(@NotNull String title, @NotNull String message, @NotNull Alert.AlertType alertType)
    {
        Timeline timeline = new Timeline(new KeyFrame(Duration.millis(1), event -> {
            Alert alert = new Alert(alertType);
            alert.initStyle(StageStyle.UTILITY);
            alert.setTitle(title);
            alert.setHeaderText(null);
            alert.setContentText(message);
            alert.show();
            if(alertType.equals(Alert.AlertType.ERROR))
                alert.setOnCloseRequest(event1 -> {
                    System.out.println("Leaving app from Error Prompt Handler......");
                    System.exit(-23);
                });
        }));
        timeline.setCycleCount(1);
        timeline.play();
    }

    public static boolean confirmationDialog(@NotNull String title, @NotNull String header, @NotNull String content)
    {
        Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
        alert.setTitle(title);
        alert.setHeaderText(header);
        alert.setContentText(content);

        Optional<ButtonType> result = alert.showAndWait();
        if (result.get() == ButtonType.OK){
            return true;
        } else {
            return false;
        }
    }
}