# Module3Capstone
Amanda Sibuea

## Main Topic: Predicting Customer Churn in E-Commerce Using Machine Learning

## Introduction
This capstone project for the Purwadhika Data Science Programme focuses on exploring data from start to finish regarding E-Commerce Customer Churn.
The goal of this project is to apply the skills nad knowledge regarding machine learning gained in Module 3 in a practival case study.
Demonstrating my understanding of key concepts and usage of tools covered in the module.

## Problem Description
The data set belongs to a leading online E-commerce company. An online retail company wants to know the customers who are going to churn, so accordingly, they can approach customers to offer some promos.
With that, the goal of this project is to create a prediction model to accurately predict the customers who will potentially churn based on the data of the features collected in the dataset provided.

### Target
0: Does not churn <br>
1: Churns

In this project, a churned customer is defined based on the label provided in the dataset.
The churn column indicates whether a customer is considered churned (1) or not-churned (0), likely based on business rules such as inactivity or purchase patterns.
Although the specific criteria used to assign this label weren’t explicitly provided, the model uses available behavioral features like Day Since Last Order, Tenure, and Complaint history, to learn patterns associated with customer churn.

### Features
- Tenure: tenure of a customer in the company
- WarehouseToHome: distance between the warehouse to the customer's home
- NumberOfDeviceRegistered: total number of devices registered on a particular customer
- PreferedOrderCat: prefered order category of a customer in the last month
- SatisfactionScore: satisfactory score of a customer on service
- MaritalStatus: marital status of a customer
- NumberOfAddress: total number of address on a particular customer
- Complain: whether a customer has raised a complain in the last month
- DaySinceLastOrder: day since last order by customer
- CashbackAmount : average cashback in last month