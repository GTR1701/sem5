USE AdventureWorks2014

SELECT *
FROM Sales.SalesOrderDetail

SELECT 
    YEAR(OrderDate) AS OrderYear,
    SUM(TotalDue) AS TotalSales
FROM 
    Sales.SalesOrderHeader
GROUP BY 
    YEAR(OrderDate)
ORDER BY 
    OrderYear ASC;

SELECT 
    YEAR(OrderDate) AS OrderYear,
    COUNT(SalesOrderID) AS OrderCount,
    SUM(TotalDue) AS TotalSales
FROM 
    Sales.SalesOrderHeader
GROUP BY 
    YEAR(OrderDate)
ORDER BY 
    OrderYear ASC;

SELECT
 YEAR(soh.OrderDate) AS SalesYear,
 SUM(soh.TotalDue) AS TotalSales
FROM Sales.SalesOrderHeader AS soh
INNER JOIN Sales.SalesOrderDetail AS sod
 ON soh.SalesOrderID = sod.SalesOrderID
GROUP BY YEAR(soh.OrderDate)
ORDER BY SalesYear;

SELECT 
    pc.Name AS CategoryName,
    SUM(sod.LineTotal) AS TotalSalesValue,
    SUM(sod.OrderQty) AS TotalQuantitySold
FROM 
    Sales.SalesOrderDetail AS sod
INNER JOIN 
    Production.Product AS p 
    ON sod.ProductID = p.ProductID
INNER JOIN 
    Production.ProductSubcategory AS psc 
    ON p.ProductSubcategoryID = psc.ProductSubcategoryID
INNER JOIN 
    Production.ProductCategory AS pc 
    ON psc.ProductCategoryID = pc.ProductCategoryID
GROUP BY 
    pc.Name
ORDER BY 
    TotalSalesValue DESC;

SELECT 
    st.Name AS TerritoryName,
    st.[Group] AS TerritoryGroup,
    SUM(soh.TotalDue) AS TotalSalesValue
FROM 
    Sales.SalesOrderHeader AS soh
INNER JOIN 
    Sales.SalesTerritory AS st 
    ON soh.TerritoryID = st.TerritoryID
GROUP BY 
    st.Name,
    st.[Group]
ORDER BY 
    TotalSalesValue DESC;

SELECT 
    CASE 
        WHEN OnlineOrderFlag = 1 THEN 'Internetowe'
        ELSE 'Nieinternetowe / Pozostałe'
    END AS OrderType,
    COUNT(*) AS OrderCount,
    SUM(TotalDue) AS TotalSalesValue,
    AVG(TotalDue) AS AverageOrderValue
FROM 
    Sales.SalesOrderHeader
GROUP BY 
    OnlineOrderFlag
ORDER BY 
    TotalSalesValue DESC;