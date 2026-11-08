USE AdventureWorks2014

SELECT ProductID, Name
FROM Production.Product;
SELECT BusinessEntityID, TerritoryID, SalesQuota
FROM Sales.SalesPerson;

SELECT *
FROM Production.Product

SELECT ProductID, Name, Color, ListPrice, ListPrice-StandardCost as Income, 
CASE Class
  WHEN 'H' THEN 'High'
  WHEN 'M' THEN 'Medium'
  WHEN 'L' THEN 'Low'
END as Class,
CASE ProductLine
  WHEN 'R' THEN 'Road'
  WHEN 'M' THEN 'Mountain'
  WHEN 'T' THEN 'Touring'
  WHEN 'S' THEN 'Standard'
END as ProductLine
FROM Production.Product
ORDER BY Name ASC

SELECT Name, ListPrice
FROM Production.Product
WHERE ListPrice > 10 AND (Color IS NULL OR Color='Red')
ORDER BY ListPrice ASC

SELECT Production.Product.Name, Production.ProductSubcategory.Name AS ProductSubcategoryName 
FROM Production.Product JOIN Production.ProductSubcategory ON Production.Product.ProductSubcategoryID=Production.ProductSubcategory.ProductSubcategoryID

SELECT Production.Product.Name, Production.ProductSubcategory.Name AS ProductSubcategoryName, Production.ProductCategory.Name AS ProductCategoryName 
FROM (Production.Product 
  JOIN Production.ProductSubcategory 
  ON Production.Product.ProductSubcategoryID=Production.ProductSubcategory.ProductSubcategoryID
  )
  JOIN Production.ProductCategory
  ON Production.ProductCategory.ProductCategoryID=Production.ProductSubcategory.ProductCategoryID

--e
SELECT 
Production.ProductCategory.Name AS ProductCategoryName, 
Production.ProductSubcategory.Name AS ProductSubcategoryName, 
AVG(Production.Product.ListPrice) as AverageListPrice, 
COUNT(Production.ProductSubcategory.ProductSubcategoryID) AS SubcategoryCount
FROM Production.Product 
  JOIN Production.ProductSubcategory 
  ON Production.Product.ProductSubcategoryID=Production.ProductSubcategory.ProductSubcategoryID
  JOIN Production.ProductCategory
  ON Production.ProductCategory.ProductCategoryID=Production.ProductSubcategory.ProductCategoryID
GROUP BY Production.ProductSubcategory.Name, Production.ProductCategory.Name
ORDER BY Production.ProductCategory.Name, Production.ProductSubcategory.Name ASC

SELECT 
    p.Name AS ProductName,
    p.ListPrice,
    AVG(p.ListPrice) OVER () AS OverallAverageListPrice,
    AVG(p.ListPrice) OVER (PARTITION BY pc.ProductCategoryID) AS CategoryAverageListPrice
FROM 
    Production.Product AS p
LEFT JOIN 
    Production.ProductSubcategory AS ps ON p.ProductSubcategoryID = ps.ProductSubcategoryID
LEFT JOIN 
    Production.ProductCategory AS pc ON ps.ProductCategoryID = pc.ProductCategoryID;

SELECT * FROM HumanResources.Employee