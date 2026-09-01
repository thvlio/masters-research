select
    di.subject_id,
    di.seq_num,
    di.icd9_code,
    p.dod,
    p.expire_flag,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    a.deathtime,
    a.admission_type,
    a.hospital_expire_flag,
    i.intime,
    i.outtime,
    did.short_title,
    did.long_title
from
    diagnoses_icd as di
        left join
            patients as p
        on
            di.subject_id = p.subject_id
        left join
            admissions as a
        on
            di.subject_id = a.subject_id
            and di.hadm_id = a.hadm_id
        left join
            icustays as i
        on
            di.subject_id = i.subject_id
            and di.hadm_id = i.hadm_id
        left join
            d_icd_diagnoses as did
        on
            di.icd9_code = did.icd9_code;

-- check if record time is between icustays intime and outtime



select
    subject_id,
    hadm_id
--    admittime,
--    dischtime,
--    intime,
--    outtime
from
(
    select
        di.subject_id,
        di.seq_num,
        di.icd9_code,
        p.dod,
        p.expire_flag,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        a.deathtime,
        a.admission_type,
        a.hospital_expire_flag,
        i.intime,
        i.outtime,
        did.short_title,
        did.long_title
    from
        diagnoses_icd as di
            left join
                patients as p
            on
                di.subject_id = p.subject_id
            left join
                admissions as a
            on
                di.subject_id = a.subject_id
                and di.hadm_id = a.hadm_id
            left join
                icustays as i
            on
                di.subject_id = i.subject_id
                and di.hadm_id = i.hadm_id
            left join
                d_icd_diagnoses as did
            on
                di.icd9_code = did.icd9_code
)
group by
    subject_id,
    hadm_id;



select
    subject_id,
    hadm_id,
    icd9_code
from
    diagnoses_icd
group by
    subject_id,
    hadm_id,
    icd9_code
having
    icd9_code like '250%';



select
    count(*)
from (
    select
        subject_id,
        hadm_id,
        icd9_code
    from
        diagnoses_icd
    group by
        subject_id,
        hadm_id,
        icd9_code
    having
        icd9_code like '250%'
);



select
    subject_id,
    hadm_id
from (
    select
        subject_id,
        hadm_id,
        icd9_code
    from
        diagnoses_icd
    group by
        subject_id,
        hadm_id,
        icd9_code
    having
        icd9_code like '250%'
        or icd9_code like '40%'
        or icd9_code like '41%'
        or icd9_code like '42%'
        or icd9_code like '43%'
--        and icd9_code not like '40%'
--        and icd9_code not like '41%'
--        and icd9_code not like '42%'
--        and icd9_code not like '43%'
)
group by
    subject_id,
    hadm_id;



select
    subject_id,
    hadm_id,
    sum(case when icd9_code like '250%' then 1 else 0 end) as diabetes,
    sum(case when icd9_code like '4_%' then 1 else 0 end) as other
from
    diagnoses_icd
where
    icd9_code like '250%'
    or icd9_code like '40%'
    or icd9_code like '41%'
    or icd9_code like '42%'
    or icd9_code like '43%'
group by
    subject_id,
    hadm_id;
