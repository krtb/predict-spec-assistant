WITH closed_opps as (
    SELECT renewal_opps.Id,
        renewal_opps.AccountId,
        case
            when renewal_opps.StageName = ('Closed Lost') then 'churn'
            when renewal_opps.StageName = ('Closed Won') then 'renewal'
        end as cutoff_type,
        DATE_ADD (
            'day',
            -30,
            LEAST (
                CloseDate,
                (
                    SELECT *
                    FROM frwd_saved_query.sync_date
                )
            )
        ) as start_date,
        --pulls 30 days prior to whichever happened first the opp close date or the sync date
        150 as random_window --
        FROM
        frwd_saved_query.renewal_opps
        JOIN salesforce.Account ON renewal_opps.AccountId = Account.Id
    WHERE 1 = 1
        AND LOWER(Account.Brand__c) in ('chrome river')
        AND renewal_opps.StageName in ('Closed Lost', 'Closed Won')
),
audience as (
    SELECT renewal_opps.Id,
        renewal_opps.AccountId,
        'audience' as cutoff_type
    FROM frwd_saved_query.renewal_opps
        JOIN salesforce.Account ON renewal_opps.AccountId = Account.Id
    WHERE 1 = 1
        AND LOWER(Account.Brand__c) in ('chrome river')
        AND renewal_opps.StageName not in ('Closed Won', 'Closed Lost')
        AND DATE_DIFF (
            'day',
            (
                SELECT *
                FROM frwd_saved_query.sync_date
            ),
            Account.std_Next_Renewal_Date__c
        ) BETWEEN 30 and 180 --renewals 30 days to 180 days away away 
)
SELECT Id,
    renewal_opps.AccountId,
    COALESCE(closed_opps.cutoff_type, audience.cutoff_type) as cutoff_type,
    case
        when closed_opps.cutoff_type = 'churn' then closed_opps.random_window
    end as churn_random_window,
    case
        when closed_opps.cutoff_type = 'churn' then closed_opps.start_date
    end as churn_start_date,
    case
        when closed_opps.cutoff_type = 'renewal' then closed_opps.random_window
    end as renewal_random_window,
    case
        when closed_opps.cutoff_type = 'renewal' then closed_opps.start_date
    end as renewal_start_date,
    (
        CASE
            WHEN closed_opps.cutoff_type = 'churn'
            and closed_opps.random_window IS NOT NULL THEN DATE_ADD (
                'day',
                - MOD(
                    ABS(
                        from_big_endian_64 (xxhash64 (CAST(Id AS varbinary)))
                    ),
                    closed_opps.random_window
                ),
                closed_opps.start_date
            )
            WHEN closed_opps.cutoff_type = 'renewal'
            and closed_opps.random_window IS NOT NULL THEN DATE_ADD (
                'day',
                - MOD(
                    ABS(
                        from_big_endian_64 (xxhash64 (CAST(Id AS varbinary)))
                    ),
                    closed_opps.random_window
                ),
                closed_opps.start_date
            )
            ELSE (
                SELECT *
                FROM frwd_saved_query.sync_date
            )
        END
    ) as cutoff_date
FROM frwd_saved_query.renewal_opps
    LEFT JOIN closed_opps USING (Id)
    LEFT JOIN audience USING (Id)