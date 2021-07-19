#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#


library(shiny)
library(leaflet)
library(knitr)
library(ggplot2)
library(tidyr)
library(caret)
library(e1071)
library(ROCR)
library(DT)
# Define UI for application that draws a histogram
shinyUI(fluidPage(

    # Application title
    titlePanel("Termo Dataset Machine Learning Algorithms"),
    tags$div(
        HTML("Anthony Cárdenas and Diego Vallejo"),
        tags$br()
       
    ),
    # Sidebar with a slider input for number of bins
    sidebarLayout(
        sidebarPanel(
            fluidRow(
               
                column(width = 12,selectInput('algoritmo', 'Algorithms' , selected="KNN", c("KNN","Random Forest","Naive Bayes","Regresión Logística","Red Neuronal","Maquina de Soporte Vectorial"))),
                column(width = 12,selectInput('date', 'Date' , selected="Enero", c("Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"))),
                column(width = 6,selectInput('ciudad', 'Shipping' , selected="Quito", c("Quito","Guayaquil","Ambato","Loja","Portoviejo","Riobamba","Cuenca"))),
                column(width = 6, selectInput('terapeutic', 'Therapeutic' , selected="Granulocyte", c("Antineoplastic","Granulocyte"))),
                column(width = 12,sliderInput('concentration','Concentration',value = 150, min = 0.3, max =300)),
                column(width = 12,sliderInput('number','Number',value = 500, min = 1, max =1218)),
                column(width = 12,sliderInput('cooler','Coolers',value = 9, min = 1, max =17)),
                column(width = 12,sliderInput('capacity','Capacity',value = 50, min = 25, max =80)),
                column(width = 12,sliderInput('temperatura','Temperature',value = 4.6, min = 4.2, max =5.0))
                              
            
                
            
        )),

        # Show a plot of the generated distribution
        mainPanel(
            
            tabsetPanel(
                tabPanel(
                    "Voting System",tableOutput("table_estimacion"),imageOutput("radar")),
                tabPanel(
                    "Predictions", DT::dataTableOutput("table"),
                    "",textOutput("acierto"),
                    "Results",textOutput("prediccion")
                ),
                tabPanel("Map",leafletOutput("my_leaf")
                         ),
                tabPanel("About",imageOutput("ia"),imageOutput("search"))
               
                
            )
            
            
            
        )
    )
))
