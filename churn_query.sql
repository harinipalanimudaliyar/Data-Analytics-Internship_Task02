-- =========================================================================
-- TASK 2: CUSTOMER CHURN DATA EXTRACTION & THRESHOLD CALCULATION
-- Objective: Identify the 90th percentile of user inactivity to define churn.
-- =========================================================================

WITH UserActivityIntervals AS (
    SELECT 
        user_id,
        signup_date,
        action_date,
        -- Calculate the exact number of days between this action and the previous one
        action_date - LAG(action_date) OVER (PARTITION BY user_id ORDER BY action_date) AS days_between_actions
    FROM virtualworks_raw_logs
)

-- 1. Extract the 90th percentile threshold (Matches the video strategy!)
SELECT 
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY days_between_actions) AS operational_churn_threshold
FROM UserActivityIntervals;

-- 2. Build the final dataset for our Python Machine Learning model
SELECT 
    user_id,
    signup_date,
    MAX(days_between_actions) AS days_since_last_action,
    COUNT(action_date) AS total_actions_6_months,
    -- Label 1 if they crossed our 90-day threshold, 0 if they stayed active
    CASE WHEN MAX(days_between_actions) >= 90 THEN 1 ELSE 0 END AS churn_label
FROM UserActivityIntervals
GROUP BY user_id, signup_date;
