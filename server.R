#
# This is the server logic of a Shiny web application. You can run the
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
library(MASS)
library(neuralnet)
library(nnet)
library(ggplot2)
library(caret)
library(shiny)
library(knitr)
library(ggplot2)
library(tidyr)
library(caret)
library(e1071)
library(ROCR)
library(class)
library(stats)
library(dplyr)
library(randomForest)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
  
  
  ### GRAFICO MAPAS
  dataFrameCiudades = data.frame(Ciudad=c('Quito','Guayaquil','Loja','Riobamba','Cuenca','Portoviejo','Ambato'), lng = c(-78.5248,-79.88621,-79.20422,-78.64712,-78.8892,-80.45445,-78.61675), lat= c(-0.225219, -2.19616,-3.99313, -1.67098,-2.71380,-1.05458,-1.24908))
  
  getColor <- function(quakes) {
    sapply(quakes$Ciudad, function(mag) {
      if(mag == "Quito") {
        "green"
      } else if(mag == "Guayaquil") {
        "orange"
      } else if(mag == "Cuenca") {
        "pink"
      } else if (mag == "Loja") {
        "red"
      }
      else if (mag == "Portoviejo") {
        "blue"
      }
      else if (mag == "Ambato") {
        "gray"
      }
      else if (mag == "Riobamba") {
        "white"
      }
      
      
      
    })
  }
  
  icons <- awesomeIcons(
    icon = 'ios-close',
    iconColor = 'black',
    library = 'ion',
    markerColor = getColor(dataFrameCiudades)
  )
  
  
  output$my_leaf <- renderLeaflet({
    
    
    leaflet(dataFrameCiudades) %>% addTiles() %>%
      addAwesomeMarkers(~lng, ~lat, icon=icons, label=~as.character(Ciudad))
  })
  
  ####################################################
  
  ## ALGORITMOS 
  termo <- read.csv(file = "Termo.csv", header = TRUE,
                    sep = ",")
  
  #######
  
  dimension = dim(termo)
  termo <- termo[,-(1),drop=FALSE]  #ID
  termo <- termo[,-(9:11),drop=FALSE] #Kinetic, Max y Min (no se conoce a-priori)
  
  #Transformacion de la variable Date de tipo char a tipo Date
  termo$Date <- as.Date(termo$Date,'%d/%m/%Y')
  termo$Date <- as.numeric(format(termo$Date,'%m')) 
  
  #Codificación: verano (Junio a Noviembre (6 a 11) => 1) e invierno (Diciembre a Mayo(12 y del 1 a 5) => 0)
  for(i in 1:dimension[1]) {
    if(termo$Date[i]== 6 || termo$Date[i]== 7 || termo$Date[i]== 8   || termo$Date[i]== 9 || termo$Date[i]== 10 || termo$Date[i]== 11 ){
      termo$Date[i] = 1
    }
    else{
      termo$Date[i] = 0
    }
  }
  
  termo$State <- (termo$State=="compliance")*1 #Codificación: cumple => 1 y no cumple => 0  
  termo$Therapeutic <- (termo$Therapeutic=="Antineoplastic")*1 #Codificación: antineoplastic => 1 y granulocyte => 0
  
  # Balancing Class (114 instancias in total)
  set.seed(12) # random
  ze = which(termo$State %in% c(0))
  le = length(ze)
  m2 = sample(ze,(round(le))) #proportion of zeros in the test dataset
  on = which(termo$State %in% c(1))
  m1 = sample(on,(round(le))) #proportion of ones in the test dataset
  m = c(m1,m2)
  ran = sample(m,length(m))
  termob <- termo[ran,]
  
  #Shipping (7 places): Ambato, Cuenca, Guayaquil, Loja, Portoviejo, Quito, Riobamba  
  termo1 <- termob[1:80,]
  termo1 <- rbind(termo1,termo[97,],termo[98,],termo[199,],termo[170,]) #Agregar instancias (ciudades) faltantes 
  termo_sol <- glm(termo1$State ~ .,family=binomial(link="logit"),data = termo1)
  
  # Test Stage 
  termo2 <- termob[85:114,]
  
  termo2 <- rbind(termo2,termo[193,],termo[200,],termo[201,],termo[203,])
  
  
  ##### FUNCION CODIFICACION CIUDAD
  
  codificacion_Ciudad <- function(numTermo) {
    
    tam_numTermo <- dim(numTermo)
    for(i in 1:tam_numTermo[1]) {
      if(numTermo$Shipping[i] == "Ambato" ){
        numTermo$Shipping[i] = "000100"
      }
      if(numTermo$Shipping[i] == "Quito" ){
        numTermo$Shipping[i] = "001000"
      }
      if(numTermo$Shipping[i] == "Guayaquil" ){
        numTermo$Shipping[i] = "001001"
      }
      
      if(numTermo$Shipping[i] == "Riobamba" ){
        numTermo$Shipping[i] = "001100"
      }
      if(numTermo$Shipping[i] == "Cuenca" ){
        numTermo$Shipping[i] = "001101"
      }
      if(numTermo$Shipping[i] == "Loja" ){
        numTermo$Shipping[i] = "001110"
      }
      if(numTermo$Shipping[i] == "Portoviejo" ){
        numTermo$Shipping[i] = "001111"
      }
    }
    numTermo  
  }
  
  
  
  
  #####################
  
  ####################### NB
   
 
  AlgoritmoNB <- function(termoTra,termoTe) {
    trainNB = termoTra
    testNB = termoTe
    
    
    NBClassificador = naiveBayes(factor(State) ~., data = trainNB)
    
    testNB$predicted = predict(NBClassificador,testNB)
    testNB$actual = testNB$State
    
    matrizConfusionNB <- confusionMatrix(factor(testNB$predicted),factor(testNB$actual))
    print(matrizConfusionNB$overall[1])
    return(matrizConfusionNB) 
  }
  
  
  
  
  ##################### KNN
  
  
  
  AlgoritmoKNN <- function(termoTra,termoTest) {
    termoTra$Shipping <- as.numeric(termoTra$Shipping) 
    termoTra <- codificacion_Ciudad(termoTra)
    
    termoTest$Shipping <- as.numeric(termoTest$Shipping) 
    termoTest <- codificacion_Ciudad(termoTest)
    
    trai <- termoTra[,-c(9)]
    test <- termoTest[,-c(9)]
    
    lab_tra <- as.factor(termoTra[,9])
    lab_test <- as.factor(termoTest[,9])
    
    predKnn <- (prediccion= knn(train = trai, test = test ,cl = lab_tra, k=2,prob=TRUE))
    confuMatrx = confusionMatrix((lab_test),(predKnn))
    return(confuMatrx)
  }
  
  ################# Random Forest
  
  
  AlgortimoRandomForest <- function(termoTra,termoTest) {
    termoTra$Shipping <- as.numeric(termoTra$Shipping) 
    termoTra <- codificacion_Ciudad(termoTra)
    
    termoTest$Shipping <- as.numeric(termoTest$Shipping) 
    termoTest <- codificacion_Ciudad(termoTest)
    
    
    training = termo1
    test = termo2
    
    modelRandomF = randomForest(State ~ .,data = training)
    
    
    predicRF = predict(modelRandomF,test)
    
    test$State_Pred = predicRF
    
    maxit = max(test$State_Pred) 
    minit = min(test$State_Pred)
    tresholdt = (minit+maxit)/2 # treshold decision
    
    tam_num = dim(test)
    for(i in 1:tam_num[1]) {
      
      if (test$State_Pred[i] >= tresholdt){
        test$State_Pred[i] = 1
      }
      else {
        test$State_Pred[i] = 0
      }
      
    }
    
    
    CFM = table(test$State,test$State_Pred)
    CFM <- confusionMatrix(factor(test$State),factor(test$State_Pred))
    
    return(CFM)
    
    
  }
  
  ######################### Regresión Logística
  
  AlgoritmoRegresionLogistica <- function(termoSol,termoTest) {
    y_estimt <- predict(termoSol,newdata=subset(termoTest,select=c(1:8)),type='response')
    
    # Logistic Regression Equation 
    maxit = max(y_estimt) 
    minit = min(y_estimt) 
    tresholdt = (minit+maxit)/2 # treshold decision
    treshold_1t = rep(c(tresholdt),each=length(y_estimt)) #vector of treshold decision
    reg_pred = (y_estimt > tresholdt)*1
    
    CmRegLog <- confusionMatrix(factor(termoTest$State),factor(reg_pred))
    
    
    return (CmRegLog) 
    
  }
  
  AlgoritmoSVM <- function(termoSol,termoTest) {
    
    svm_model <- train(State ~ .,
                       data = termoSol,
                       method = "svmLinear",
                       tuneLength= 20,
                       preProcess = c("center", "scale"))
    
    
    pred <- predict(svm_model, termoTest)
    
    max <- max(pred)
    min <- min(pred)
    
    tresh <- (max+min)/2
    
    dimension <- length(pred)
    for(i in 1:dimension) {
      
      if (pred[i] >= tresh){
        pred[i] = 1
      }
      else {
        pred[i] = 0
      }
    }
    
    CmSVM <- confusionMatrix(factor(termo2$State),factor(pred))
    
    return (CmSVM)
  }
  
  ################# RED NEURONAL
  
  
  AlgoritmoRedNeuronal <- function(termoOriginal,termoTrain,termoTest) {
    
    termoTrain$Shipping <- as.numeric(termoTrain$Shipping) 
    termoTrain <- codificacion_Ciudad(termoTrain)
    
    termoTest$Shipping <- as.numeric(termoTest$Shipping) 
    termoTest <- codificacion_Ciudad(termoTest)
    
    ra <- apply(termoOriginal,2,max)
    
    r <- 1/1218
    
    
    mod <- nnet(State ~ ., data = termoTrain,
                size=3, maxit=10000,
                decay = .001, rang = r,
                na.action = na.omit, skip=TRUE)
    
    pred <- (predict(mod, newdata = termoTest))
    
    maxit <- max(pred) 
    minit <- min(pred) 
    tresholdt <- (minit+maxit)/2
    
    reg <- data.frame(as.integer((pred > tresholdt)*1))
    
    real <- data.frame((termo2[9]))
    b <- data.frame(reg, real)
    names (b) = c("predicho", "real")
    
    a = confusionMatrix(factor(b$predicho),factor(b$real))
    
    
    
    return (a)
  }
  
  
  
  
  ########################3
  
  ###################################################
  
  ###  FUNCIONES PARA CALCULAR CON LOS VALORES SELECCIONADOS POR EL USUARIO
  

  
  
  ################# Random Forest
  
  
  AlgoritmoRandomForestPred <- function(termoTra,termoTest,termo3) {
    termoTra$Shipping <- as.numeric(termoTra$Shipping) 
    
    termoTest$Shipping <- as.factor(termoTest$Shipping) 
    
    termo3$Shipping <- as.factor(termo3$Shipping)
    
    training = termo1
    test = termo2
    
    modelRandomF = randomForest(State ~ .,data = training)
    
    
    predicRF = predict(modelRandomF,test[-9])
    
    test$State_Pred = predicRF
    
    maxit = max(test$State_Pred) 
    minit = min(test$State_Pred)
    tresholdt = (minit+maxit)/2 # treshold decision
  
    ter <- termoTest
    ter[1,] = termo3    
    predicRF = predict(modelRandomF,ter[1,])
  
    reg <- (as.integer((predicRF > tresholdt)*1))
    
    if (reg == 1){
      prediccion <- "Cumple"
    }else {
      prediccion <- "No cumple"
    }
    return (prediccion)
    
  }
  
  ######################### Naive Bayes
  AlgoritmoNBPred <- function(termoTra,termoTe,termo3) {
    trainNB = termoTra
    testNB = termoTe
    
    termo3$Shipping <- as.factor(termo3$Shipping)
    
    NBClassificador = naiveBayes(factor(State) ~., data = trainNB)
    
    testNB$predicted = predict(NBClassificador,testNB)
    testNB$actual = testNB$State
  
    
    
    ter <- termoTe
    ter[1,] = termo3    
    
    testUsu = predict(NBClassificador,ter[1,])
    
    if (testUsu == 1) {
      prediccion = "Cumple"
    }else {
      prediccion = "No cumple"
    }
    return(prediccion)
  }
  
  
  ################ Regresión Logística
  AlgoritmoRegresionLogisticaPred <- function(termo_sol,termoTest,termo3) {
    y_estimt <- predict(termo_sol,newdata=subset(termoTest,select=c(1:8)),type='response')
    
    # Logistic Regression Equation 
    maxit = max(y_estimt) 
    minit = min(y_estimt) 
    tresholdt = (minit+maxit)/2 # treshold decision
    
    
    
    y_estimt <- predict(termo_sol,newdata=subset(termo3,select=c(1:8)),type='response')
    
    if (y_estimt > tresholdt) {
      prediccion <- "Cumple"
    }else {
      prediccion <- "No cumple"
    }
    return (prediccion)
    }
  
  
  ########### SVM Algoritmo 
  AlgoritmoSVMPred <- function(termo1,termo2,termo3) {
    svm_model <- train(State ~ .,
                       data = termo1,
                       method = "svmLinear",
                       tuneLength= 20,
                       preProcess = c("center", "scale"))
    
    
    pred <- predict(svm_model, termo2)
    
    max <- max(pred)
    min <- min(pred)
    
    tresh <- (max+min)/2
    
    print("----")
    print(tresh)
    print("----")
    pred <- predict(svm_model, termo3)
    print(pred)
    if (pred > tresh) {
      prediccion <- "Cumple"
    }else {
      prediccion <- "No cumple"
      
    }
    return (prediccion)
  } 
  
  ####################### RED NEURONAL 
  
  AlgoritmoRedNeuronalPred <- function (termoTrain,termoTest,termo3) {
    termoTrain$Shipping <- as.numeric(termoTrain$Shipping) 
    termoTrain <- codificacion_Ciudad(termoTrain)
    
    termoTest$Shipping <- as.numeric(termoTest$Shipping) 
    termoTest <- codificacion_Ciudad(termoTest)
    
    termo3$Shipping <- as.numeric(termo3$Shipping) 
    termo3 <- codificacion_Ciudad(termo3)
    
    r <- 1/1218
    
    
    mod <- nnet(State ~ ., data = termoTrain,
                size=3, maxit=10000,
                decay = .001, rang = r,
                na.action = na.omit, skip=TRUE)
    
    pred <- (predict(mod, newdata = termoTest))
    
    maxit <- max(pred) 
    minit <- min(pred) 
    tresholdt <- (minit+maxit)/2
    
    pred <- (predict(mod, newdata = termo3))
    print(tresholdt)
    print("----")
    print(pred)
    if (pred > tresholdt) {
      prediccion <- "Cumple"
    }else {
      prediccion <- "No cumple"
    }
    return (prediccion)
  }
  
  ####################### KNN
  
  AlgoritmoKNNPred <- function(termo1,termo3) {
    termo1$Shipping <- as.numeric(termo1$Shipping) 
    termo1 <- codificacion_Ciudad(termo1)
    
    trai <- termo1[,-c(9)]
    lab_tra <- as.factor(termo1[,9])
    
    
    predKnn <- as.numeric(knn(train = trai, test = termo3 ,cl = lab_tra, k=2,prob=TRUE))
    
    if(predKnn == 2) {
      prediccion = "Cumple"
    }
    else {
      prediccion = "No cumple"
    }
    return (prediccion)
  }
  

  
  
  
  #################################################
  
  ################################################
  
  output$acierto <- renderText({
   
   
    if (input$algoritmo == "KNN"){
      
      dataFrameKNN <- estadisticaTable(termo1,termo2,termo_sol)
      output$table <- DT::renderDataTable({
        DT::datatable(dataFrameKNN)
      })
     a <- AlgoritmoKNN(termo1,termo2)
     a$overall[1]
     ""
     }
    else  if (input$algoritmo == "Naive Bayes"){
      
      dataFrameNB <- estadisticaTable(termo1,termo2,termo_sol)
      output$table <- DT::renderDataTable({
        DT::datatable(dataFrameNB)
      })
      a <- AlgoritmoNB(termo1,termo2)
      a$overall[1]
      "" 
      
      
      
      }
    else if (input$algoritmo == "Random Forest"){
      dataFrameRF <- estadisticaTable(termo1,termo2,termo_sol)
      output$table <- DT::renderDataTable({
        DT::datatable(dataFrameRF)
      }) 
      
    a <- AlgortimoRandomForest(termo1,termo2)
     a$overall[1]
     ""
     }
    else if (input$algoritmo == "Regresión Logística"){
      
      dataFrameRL <- estadisticaTable(termo1,termo2,termo_sol)
      output$table <- DT::renderDataTable({
        DT::datatable(dataFrameRL)
      })
      a <- AlgoritmoRegresionLogistica(termo_sol,termo2)
      a$overall[1]
      ""
      }
    else if (input$algoritmo == "Maquina de Soporte Vectorial"){
      dataFrameSVM <- estadisticaTable(termo1,termo2,termo_sol)
      output$table <- DT::renderDataTable({
        DT::datatable(dataFrameSVM)
      }) 
      
    a <- AlgoritmoSVM(termo1,termo2)
     a$overall[1]
     ""
     
      }
    else if (input$algoritmo == "Red Neuronal") {
      dataFrameRedN <- estadisticaTable(termo1,termo2,termo_sol)
      output$table <- DT::renderDataTable({
        DT::datatable(dataFrameRedN)
      })
      
      a <- AlgoritmoRedNeuronal(termo,termo1,termo2)
     a$overall[1]
     ""
     
     }
    
    
  })
  
  
  
  #################################
  
  
  
  
  #########################################
  
  
  
  
  ########################### Output Prediccion Algoritmos
  
  
  
  output$prediccion <- renderText({
    
    
    if (input$algoritmo == "KNN"){
      
      termo3 <- data.frame(
        "Therapeutic" = input$terapeutic, 
        "Concentration" = input$concentration, 
        "Shipping" = input$ciudad,
        "Number" = input$number,
        "Coolers" = input$cooler,
        "Capaciy" = input$capacity ,
        "Temperature" = input$temperatura,
        "Date" = input$date
      )
      termo3$Therapeutic <- (termo3$Therapeutic=="Antineoplastic")*1 #CodificaciÃ³n: antineoplastic => 1 y granulocyte => 0
      
      if(termo3$Date[1]== "Diciembre" || termo3$Date[1]== "Enero" || termo3$Date[1]== "Febrero" || termo3$Date[1]== "Marzo" || termo3$Date[1]== "Abril" || termo3$Date[1]== "Mayo"){
        termo3$Date = "0"
      }else{
        termo3$Date = "1"
      }
      
      termo3$Date <- as.numeric(termo3$Date)
      
      termo3$Shipping <- as.numeric(termo3$Shipping) 
      termo3 <- codificacion_Ciudad(termo3)
      
      
      
      AlgoritmoKNNPred(termo1,termo3)
    }
    else  if (input$algoritmo == "Naive Bayes"){
      
      
      termo3 <- data.frame(
        "Therapeutic" = input$terapeutic, 
        "Concentration" = input$concentration, 
        "Shipping" = input$ciudad,
        "Number" = input$number,
        "Coolers" = input$cooler,
        "Capaciy" = input$capacity ,
        "Temperature" = input$temperatura,
        "Date" = input$date
      )
      
      termo3$Therapeutic <- (termo3$Therapeutic=="Antineoplastic")*1 #CodificaciÃ³n: antineoplastic => 1 y granulocyte => 0
      
      if(termo3$Date[1]== "Diciembre" || termo3$Date[1]== "Enero" || termo3$Date[1]== "Febrero" || termo3$Date[1]== "Marzo" || termo3$Date[1]== "Abril" || termo3$Date[1]== "Mayo"){
        termo3$Date = "0"
      }else{
        termo3$Date = "1"
      }
      
      termo3$Date <- as.numeric(termo3$Date)
      
      termo3$Shipping <- as.factor(termo3$Shipping) 
      
      
      
      AlgoritmoNBPred(termo1,termo2,termo3)
    }
    else if (input$algoritmo == "Random Forest"){
      
      
      termo3 <- data.frame(
        "Therapeutic" = input$terapeutic, 
        "Concentration" = input$concentration, 
        "Shipping" = input$ciudad,
        "Number" = input$number,
        "Coolers" = input$cooler,
        "Capaciy" = input$capacity ,
        "Temperature" = input$temperatura,
        "Date" = input$date
      )
      
      termo3$Therapeutic <- (termo3$Therapeutic=="Antineoplastic")*1 #CodificaciÃ³n: antineoplastic => 1 y granulocyte => 0
      
      if(termo3$Date[1]== "Diciembre" || termo3$Date[1]== "Enero" || termo3$Date[1]== "Febrero" || termo3$Date[1]== "Marzo" || termo3$Date[1]== "Abril" || termo3$Date[1]== "Mayo"){
        termo3$Date = "0"
      }else{
        termo3$Date = "1"
      }
      
      termo3$Date <- as.numeric(termo3$Date)
      
      termo3$Shipping <- as.factor(termo3$Shipping) 
      
      
      
      
      AlgoritmoRandomForestPred(termo1,termo2,termo3)
    }
    else if (input$algoritmo == "Regresión Logística"){
    
      termo3 <- data.frame(
        "Therapeutic" = input$terapeutic, 
        "Concentration" = input$concentration, 
        "Shipping" = input$ciudad,
        "Number" = input$number,
        "Coolers" = input$cooler,
        "Capaciy" = input$capacity ,
        "Temperature" = input$temperatura,
        "Date" = input$date
      )
      
      termo3$Therapeutic <- (termo3$Therapeutic=="Antineoplastic")*1 #CodificaciÃ³n: antineoplastic => 1 y granulocyte => 0
      
      if(termo3$Date[1]== "Diciembre" || termo3$Date[1]== "Enero" || termo3$Date[1]== "Febrero" || termo3$Date[1]== "Marzo" || termo3$Date[1]== "Abril" || termo3$Date[1]== "Mayo"){
        termo3$Date = "0"
      }else{
        termo3$Date = "1"
      }
      
      termo3$Date <- as.numeric(termo3$Date)
      
      termo3$Shipping <- as.factor(termo3$Shipping) 
      
      
      
      AlgoritmoRegresionLogisticaPred(termo_sol,termo2,termo3)
    }
    else if (input$algoritmo == "Maquina de Soporte Vectorial"){
      termo3 <- data.frame(
        "Therapeutic" = input$terapeutic, 
        "Concentration" = input$concentration, 
        "Shipping" = input$ciudad,
        "Number" = input$number,
        "Coolers" = input$cooler,
        "Capaciy" = input$capacity ,
        "Temperature" = input$temperatura,
        "Date" = input$date
      )
      
      termo3$Therapeutic <- (termo3$Therapeutic=="Antineoplastic")*1 #CodificaciÃ³n: antineoplastic => 1 y granulocyte => 0
      
      if(termo3$Date[1]== "Diciembre" || termo3$Date[1]== "Enero" || termo3$Date[1]== "Febrero" || termo3$Date[1]== "Marzo" || termo3$Date[1]== "Abril" || termo3$Date[1]== "Mayo"){
        termo3$Date = "0"
      }else{
        termo3$Date = "1"
      }
      
      termo3$Date <- as.numeric(termo3$Date)
      
      termo3$Shipping <- as.factor(termo3$Shipping) 
      
      
      
        AlgoritmoSVMPred(termo1,termo2,termo3)
    }
    else if (input$algoritmo == "Red Neuronal"){
      termo3 <- data.frame(
        "Therapeutic" = input$terapeutic, 
        "Concentration" = input$concentration, 
        "Shipping" = input$ciudad,
        "Number" = input$number,
        "Coolers" = input$cooler,
        "Capaciy" = input$capacity ,
        "Temperature" = input$temperatura,
        "Date" = input$date
      )
      
      termo3$Therapeutic <- (termo3$Therapeutic=="Antineoplastic")*1 #CodificaciÃ³n: antineoplastic => 1 y granulocyte => 0
      
      if(termo3$Date[1]== "Diciembre" || termo3$Date[1]== "Enero" || termo3$Date[1]== "Febrero" || termo3$Date[1]== "Marzo" || termo3$Date[1]== "Abril" || termo3$Date[1]== "Mayo"){
        termo3$Date = "0"
      }else{
        termo3$Date = "1"
      }
      
      termo3$Date <- as.numeric(termo3$Date)
      
      termo3$Shipping <- as.factor(termo3$Shipping) 
      
      AlgoritmoRedNeuronalPred(termo1,termo2,termo3)
      
      
    }
    
    
  })
    
  estadisticaTable <- function(termo1,termo2,termo_sol){
    
    
    
    matrizKnn <- AlgoritmoKNN(termo1,termo2)
    matrizNaibeBayes <- AlgoritmoNB(termo1,termo2)
    matrizRF <- AlgortimoRandomForest(termo1,termo2)
    matrizRL <- AlgoritmoRegresionLogistica(termo_sol,termo2)
    matrizSVM <- AlgoritmoSVM(termo1,termo2)
    matrizRedN <- AlgoritmoRedNeuronal(termo,termo1,termo2)
    
    
    
    diamonds = data.frame("Accuracy" = c(matrizKnn$overall[1],matrizNaibeBayes$overall[1],matrizRF$overall[1],matrizRL$overall[1],matrizSVM$overall[1],matrizRedN$overall[1]),
                          "Precision" = c(matrizKnn$byClass[5],matrizNaibeBayes$byClass[5],matrizRF$byClass[5],matrizRL$byClass[5],matrizSVM$byClass[5],matrizRedN$byClass[5]),
                          "Recall" = c(matrizKnn$byClass[6],matrizNaibeBayes$byClass[6],matrizRF$byClass[6],matrizRL$byClass[6],matrizSVM$byClass[6],matrizRedN$byClass[6]),
                          "F1" = c(matrizKnn$byClass[7],matrizNaibeBayes$byClass[7],matrizRF$byClass[7],matrizRL$byClass[7],matrizSVM$byClass[7],matrizRedN$byClass[7]),
                          "ROC" = c(0.5970696,0.6586538,0.62814,0.6453,0.5916117,0.63485)) 
    
    Nombres1 = c("KNN","Naive Bayes","Random Forest", "Regresión Logística","SVM","Red Neuronal")
    NewPS <- data.frame(diamonds, row.names = Nombres1)
    NewPS <- round(NewPS,4)
    return (NewPS)
    
    
  }
  
  
  
  output$table_estimacion  <- renderTable({
    
    
    
    termo3 <- data.frame(
      "Therapeutic" = input$terapeutic, 
      "Concentration" = input$concentration, 
      "Shipping" = input$ciudad,
      "Number" = input$number,
      "Coolers" = input$cooler,
      "Capaciy" = input$capacity ,
      "Temperature" = input$temperatura,
      "Date" = input$date
    )
    termo3$Therapeutic <- (termo3$Therapeutic=="Antineoplastic")*1 #CodificaciÃ³n: antineoplastic => 1 y granulocyte => 0
    
    if(termo3$Date[1]== "Diciembre" || termo3$Date[1]== "Enero" || termo3$Date[1]== "Febrero" || termo3$Date[1]== "Marzo" || termo3$Date[1]== "Abril" || termo3$Date[1]== "Mayo"){
      termo3$Date = "0"
    }else{
      termo3$Date = "1"
    }
    
    termo3$Date <- as.numeric(termo3$Date)
    
    termo3$Shipping <- as.numeric(termo3$Shipping) 
    
    termo3 <- codificacion_Ciudad(termo3)
    
    KnnPr <- AlgoritmoKNNPred(termo1,termo3)
    
    
    ##########################################
    
    
    
    NbPr <- AlgoritmoNBPred(termo1,termo2,termo3)
    
    
    ######################
    
    termo3 <- data.frame(
      "Therapeutic" = input$terapeutic, 
      "Concentration" = input$concentration, 
      "Shipping" = input$ciudad,
      "Number" = input$number,
      "Coolers" = input$cooler,
      "Capaciy" = input$capacity ,
      "Temperature" = input$temperatura,
      "Date" = input$date
    )
    
    termo3$Therapeutic <- (termo3$Therapeutic=="Antineoplastic")*1 #CodificaciÃ³n: antineoplastic => 1 y granulocyte => 0
    
    if(termo3$Date[1]== "Diciembre" || termo3$Date[1]== "Enero" || termo3$Date[1]== "Febrero" || termo3$Date[1]== "Marzo" || termo3$Date[1]== "Abril" || termo3$Date[1]== "Mayo"){
      termo3$Date = "0"
    }else{
      termo3$Date = "1"
    }
    
    termo3$Date <- as.numeric(termo3$Date)
    
    termo3$Shipping <- as.factor(termo3$Shipping) 
    
    
    SVMPr <- AlgoritmoSVMPred(termo1,termo2,termo3)
    RedNPr <- AlgoritmoRedNeuronalPred(termo1,termo2,termo3)
    
    
    
    RFPr <- AlgoritmoRandomForestPred(termo1,termo2,termo3)
    
    
        
    
    RLPr <- AlgoritmoRegresionLogisticaPred(termo_sol,termo2,termo3)
    
    
    
    datos <- data.frame("KNN" = c(KnnPr),
                        "Naibe Bayes" = c(NbPr),
                        "Random Forest" = c(RFPr),
                        "Regresión Logística" = c(RLPr),
                        "Red Neuronal" = c(RedNPr),
                        "SVM" = c(SVMPr)
                         )
        
    real <- (datos != "No cumple")*1
    
    suma <- sum(real)
    if (suma >= 4){
      resultado <- "Cumple"
    }else {
      resultado <- "No cumple"
      
    }
    datos <- data.frame("KNN" = c(real[1]),
                        "Naive Bayes" = c(real[2]),
                        "Random Forest" = c(real[3]),
                        "Regresión Logística" = c(real[4]),
                        "Red Neuronal" = c(real[5]),
                        "SVM" = c(real[6]),
                        "Algorithm Voting" = c(resultado)
    )
    datos
    
  })
  
 output$ia <- renderImage({
     list(src = "logo_ai_ingles.png",
          width = 450,
          height = 200,
          alt = "IA",
          align = "right"
     )
   
 })
 
 output$search <- renderImage({
   
   list(src = "ingles.png",
        width = 450,
        height = 200,
        alt = "Search",
        align = "left"
   )
   
 })
 

 output$radar <- renderImage({
   list(src = "radarPlotVF.png",
        width = 450,
        height = 200,
        alt = "Radar",
        align = "Center"
   )
   
 }) 
  
 
})
