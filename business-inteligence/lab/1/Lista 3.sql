use AdventureWorks2014

SELECT *
FROM Person.BusinessEntityContact

SELECT 
  SUM(LineTotal) as "Sales Amount", 
  SUM(sod.OrderQty) as "Volume",
  COUNT(distinct soh.SalesOrderID) as "Number of Orders"
FROM 
  Sales.SalesOrderHeader as soh
INNER JOIN 
  Sales.SalesOrderDetail as sod on soh.SalesOrderID = sod.SalesOrderID

SELECT 
  YEAR(soh.OrderDate) as Year,
  SUM(LineTotal) as "Sales Amount", 
  SUM(sod.OrderQty) as "Volume",
  COUNT(distinct sod.SalesOrderID) as "Number of Orders"
FROM 
  Sales.SalesOrderHeader as soh
inner JOIN 
  Sales.SalesOrderDetail as sod on
  sod.SalesOrderID = soh.SalesOrderID
GROUP BY YEAR(soh.OrderDate)
ORDER BY YEAR(soh.OrderDate)

SELECT CustomerID
FROM
  Sales.Customer as c
INNER JOIN 
  Person.Person as p
ON
  c.CustomerID = p.PersonID