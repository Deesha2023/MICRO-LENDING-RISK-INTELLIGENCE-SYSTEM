-- Micro-Lending Business Queries
-- Database: loan_database | Table: loan_sample

USE loan_database;

-- 1. Portfolio size, exposure and overall default rate
SELECT COUNT(*) AS total_loans,
       ROUND(SUM(loan_amnt), 2) AS total_amount,
       ROUND(AVG(loan_amnt), 2) AS average_loan_amount,
       ROUND(100 * AVG(CASE WHEN LOWER(TRIM(loan_condition)) = 'bad loan'
                            THEN 1 ELSE 0 END), 2) AS bad_loan_rate_pct
FROM loan_sample;

-- 2. Which credit grades have the highest bad-loan rate?
SELECT grade, COUNT(*) AS loans, ROUND(AVG(loan_amnt), 2) AS avg_loan_amount,
       ROUND(100 * AVG(CASE WHEN LOWER(TRIM(loan_condition)) = 'bad loan'
                            THEN 1 ELSE 0 END), 2) AS bad_loan_rate_pct
FROM loan_sample GROUP BY grade ORDER BY bad_loan_rate_pct DESC;

-- 3. Which loan purposes create the most risk and exposure?
SELECT purpose, COUNT(*) AS loans, ROUND(SUM(loan_amnt), 2) AS total_amount,
       ROUND(100 * AVG(CASE WHEN LOWER(TRIM(loan_condition)) = 'bad loan'
                            THEN 1 ELSE 0 END), 2) AS bad_loan_rate_pct
FROM loan_sample GROUP BY purpose ORDER BY bad_loan_rate_pct DESC;

-- 4. Geographic risk (minimum 25 loans avoids tiny-sample conclusions)
SELECT addr_state, COUNT(*) AS loans,
       ROUND(100 * AVG(CASE WHEN LOWER(TRIM(loan_condition)) = 'bad loan'
                            THEN 1 ELSE 0 END), 2) AS bad_loan_rate_pct
FROM loan_sample GROUP BY addr_state HAVING COUNT(*) >= 25
ORDER BY bad_loan_rate_pct DESC;

-- 5. Does the average interest rate differ for good and bad loans?
SELECT loan_condition, COUNT(*) AS loans,
       ROUND(AVG(CAST(REPLACE(int_rate, '%', '') AS DECIMAL(10,2))), 2)
           AS average_interest_rate_pct
FROM loan_sample GROUP BY loan_condition;

-- 6. Default performance by debt-to-income band
SELECT CASE WHEN dti < 10 THEN 'Low (<10)'
            WHEN dti < 20 THEN 'Medium (10-19.99)'
            WHEN dti < 30 THEN 'High (20-29.99)'
            ELSE 'Very High (30+)' END AS dti_band,
       COUNT(*) AS loans,
       ROUND(100 * AVG(CASE WHEN LOWER(TRIM(loan_condition)) = 'bad loan'
                            THEN 1 ELSE 0 END), 2) AS bad_loan_rate_pct
FROM loan_sample WHERE dti IS NOT NULL GROUP BY dti_band ORDER BY MIN(dti);

-- 7. Loan-term performance
SELECT term, COUNT(*) AS loans, ROUND(AVG(loan_amnt), 2) AS avg_loan_amount,
       ROUND(100 * AVG(CASE WHEN LOWER(TRIM(loan_condition)) = 'bad loan'
                            THEN 1 ELSE 0 END), 2) AS bad_loan_rate_pct
FROM loan_sample GROUP BY term ORDER BY bad_loan_rate_pct DESC;

-- 8. Top 100 largest bad loans for management review
SELECT loan_amnt, annual_inc, int_rate, dti, grade, purpose
FROM loan_sample WHERE LOWER(TRIM(loan_condition)) = 'bad loan'
ORDER BY loan_amnt DESC LIMIT 100;
