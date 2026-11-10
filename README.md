# digital-markers

`digital-markers` is an open-source Python library from the CASAS Lab at
Washington State University. The package computes marker families from
wearable sensor data: exertion, gait, location, and behavioral routine.
Activity-label and audio markers are also available.

## Supported markers

### Exertion

`compute_exertion_markers` computes daily motion-only and
heart-rate-plus-motion exertion markers, with participant-level summaries.

### Gait

`compute_gait_markers` computes daily wrist-derived walking-bout markers,
including step counts, cadence, step intervals, gait frequency, and signal
measures.

### Location

`compute_location_markers` returns participant-level and daily location
measurements, plus home-location and staypoint tables in `details`.

### Behavioral routine

`compute_routine_markers` computes model-free routine markers from 10 Hz motion
data using fixed acceleration thresholds. Its details include five-minute
interval, state-balance, region, and diagnostic tables.

### Activity

`compute_activity_markers` summarizes the labels in a separate 10 Hz activity
stream. Daily markers include time in each observed activity, activity bouts,
transitions, entropy, and time-of-day distributions; participant markers
summarize the daily values.

### Audio

`compute_audio_markers` derives acoustic, transcript, and optional behavior and
health markers from a participant's audio recordings. See the audio-specific
input, usage, language configuration, and result guidance below.

## Preparing participant data

The `compute_*_markers` functions accept a `ParticipantData` instance and
return a `MarkerResult`. Prepare one DataFrame for each sensor stream and use
the canonical column names below. Sensor timestamps must be parseable as
datetimes. Exertion requires the motion timestamp and, when heart rate is
provided, the heart-rate timestamp to have a Pandas datetime dtype. Gait and
routine require a datetime motion timestamp. Activity requires a separate
activity DataFrame with datetime `timestamp` and `activity` columns.

The package uses Pandas readers, so inputs can come from CSV or Parquet. This
CSV example assumes timestamps have no timezone or UTC offset and already
represent local wall-clock time:

```python
import pandas as pd
from digital_markers import ParticipantData

motion = pd.read_csv("motion.csv")
motion["timestamp"] = pd.to_datetime(motion["timestamp"])

heart_rate = pd.read_csv("heart_rate.csv")
heart_rate["timestamp"] = pd.to_datetime(heart_rate["timestamp"])

location = pd.read_csv("location.csv")
location["timestamp"] = pd.to_datetime(location["timestamp"])

activity = pd.read_csv("activity.csv")
activity["timestamp"] = pd.to_datetime(activity["timestamp"])

participant = ParticipantData(
    participant_id="P001",
    timezone="America/Los_Angeles",
    motion=motion,
    heart_rate=heart_rate,
    location=location,
    activity=activity,
)
```

For Parquet, use `pd.read_parquet("motion.parquet")` (and likewise for the
other streams; Pandas needs an optional Parquet engine such as `pyarrow`).
Parquet timestamps are often already datetimes; if a timestamp column was
stored as text, convert it with `pd.to_datetime` before calling the marker
functions. Do not infer a timezone from a naive timestamp: establish from the
source whether it represents local time or UTC. A naive local timestamp parsed
with `utc=True` is interpreted as UTC, not local time.

Use these column names:

| Stream | Required columns | Optional columns |
|---|---|---|
| `motion` | `timestamp`, `acceleration_x`, `acceleration_y`, `acceleration_z` | `rotation_rate_x`, `rotation_rate_y`, `rotation_rate_z`; routine can also use `latitude`, `longitude` |
| `heart_rate` | `timestamp`, `heart_rate` | None |
| `location` | `timestamp`, `latitude`, `longitude` | None |
| `activity` | `timestamp`, `activity` | None |

`ParticipantData` automatically canonicalizes sensor column names when it is
created. For example, `stamp` is accepted as `timestamp`, and
`user_acceleration_x`, `user_acceleration_y`, and `user_acceleration_z` are
accepted as the corresponding `acceleration_x`, `acceleration_y`, and
`acceleration_z` columns. Activity labels must be supplied in the `activity`
column. The input DataFrames are copied during this normalization; the original
frames are unchanged.

Use numeric values for sensor measurements. Exertion requires all three
rotation-rate columns. Gait can run without rotation-rate columns; when
provided, supply all three. Routine can run without them, but if any rotation
column is present, all three are required. Location requires the latitude and
longitude columns, but their values may be missing on rows without a GPS fix.
Keep those rows and their timestamps: location uses them to estimate observed
recording time. Routine uses `location` coordinates when supplied, or motion
coordinates when present. Its motion input is expected at 10 Hz.

### Audio

Audio recordings are attached to `ParticipantData.audio` as
`AudioRecording` instances. To load files, use `read_audio_directory`, which
scans only files directly inside the specified directory and retains their
original filenames:

```python
from digital_markers import AudioRecording, ParticipantData, read_audio_directory

recordings = read_audio_directory("recordings/P001")
participant = ParticipantData(participant_id="P001", audio=recordings)
```

Alternatively, pass in-memory mono or `(samples, channels)` audio. Supply the
sample rate in Hz. Integer PCM is normalized automatically; floating-point
samples should already be normalized to [-1, 1].

```python
recording = AudioRecording("journal.wav", samples=audio, sample_rate_hz=16000)
participant = ParticipantData(participant_id="P001", audio=[recording])
```

### Timestamp and local-time guidance

Use the participant's IANA timezone (for example, `America/Los_Angeles`) as
the source of truth for local time. Keep original timestamps and their source
timezone/UTC-offset information in your stored data whenever possible. A
timezone-naive timestamp does not identify a timezone: treat it as local only
when the source confirms it is already local. If naive timestamps represent
UTC, localize them as UTC before converting; do not parse them as local time.
For example, for a CSV column known to contain UTC timestamps:

```python
frame = pd.read_csv("motion.csv")
local_timezone = "America/Los_Angeles"
timestamps_utc = pd.to_datetime(frame["timestamp"], utc=True)
frame["timestamp"] = timestamps_utc.dt.tz_convert(local_timezone)
```

`utc=True` is appropriate in that example only because the source is known to
be UTC (or contains explicit offsets). For naive local timestamps, use
`pd.to_datetime` without `utc=True`, as in the earlier example. If localizing
naive timestamps to a named timezone, choose an explicit policy for daylight
saving transitions: a fall-back clock change can make a local time occur twice,
and a spring-forward change can make some local times nonexistent. Pandas
defaults to raising on ambiguous or nonexistent times; resolve them using
source metadata rather than guessing.

The current marker functions do not apply one common local-time conversion:

- Routine converts timezone-aware motion and location timestamps to
  `ParticipantData.timezone`, or UTC when `timezone` is omitted. Naive
  timestamps are left unchanged and treated as local clock values.
- Activity converts timezone-aware activity timestamps to
  `ParticipantData.timezone`, or UTC when `timezone` is omitted. Naive
  timestamps are left unchanged and treated as local clock values.
- Exertion, gait, and location do not use `ParticipantData.timezone` to
  convert inputs. Their daily grouping follows the timestamps as processed by
  each implementation; timezone-aware inputs may therefore not align with
  routine's local-day grouping.

Set `ParticipantData.timezone` for routine and activity and ensure that motion,
activity, and heart-rate timestamps use compatible conventions for their
respective marker functions. If local-day alignment across all marker families
is essential, convert the streams
consistently before calling them and validate results around midnight and
daylight-saving transitions. Converting timezone-aware values to local time and
then removing timezone information can align ordinary local-day boundaries, but
it loses the distinction between the two occurrences of a repeated fall-back
time; routine removes duplicate motion timestamps. Retain the original
timezone-aware timestamps so that this ambiguity can be detected and handled
for the study's data.

## Calling the marker functions

Import each function from its marker package. All functions take the same
participant object:

```python
from digital_markers.exertion import compute_exertion_markers
from digital_markers.gait import compute_gait_markers
from digital_markers.location import compute_location_markers
from digital_markers.routine import compute_routine_markers
from digital_markers.activity import compute_activity_markers
from digital_markers import compute_audio_markers

exertion = compute_exertion_markers(participant)
gait = compute_gait_markers(participant)
location_result = compute_location_markers(participant)
routine = compute_routine_markers(participant)
activity = compute_activity_markers(participant)
audio_result = compute_audio_markers(participant)
```

Exertion requires `motion`; `heart_rate` is optional and can be omitted from
`ParticipantData` or set to `None`. Gait and routine require `motion`.
Location requires `location`; `motion` is optional and contributes timestamps
to its recorded-time calculations. Routine can use location data but does not
require it. Activity requires the separate `activity` stream with `timestamp`
and `activity` columns; it does not require motion, acceleration, or other
sensor measurements. Its sampling assumption is 10 Hz. Location and routine
currently take no marker-specific keyword arguments; routine configuration
uses fixed thresholds.

### Optional audio arguments

Install the optional audio dependencies with `pip install digital-markers[audio]`.
`compute_audio_markers` uses Whisper's automatic language detection and device
selection by default. Audio computation does not write files as a side effect.

| Argument | Default | Purpose |
|---|---:|---|
| `language` | `"auto"` | Whisper language code, or `"auto"` to detect the language. |
| `model` | `"large-v3"` | Faster-Whisper model name. |
| `device` | `"auto"` | Inference device: `"auto"`, `"cpu"`, or `"cuda"`. |
| `beam_size` | `5` | Whisper beam size. |
| `min_words` | `8` | Minimum transcript word count for informative text markers. |
| `include_behavior` | `False` | Include behavior marker columns. |
| `include_health` | `False` | Include health marker columns. |
| `text_patterns` | `None` | Language-specific regular expressions for activity, health, and specificity markers. |
| `syntax_models` | `None` | Mapping of language codes to spaCy model names for syntax markers. |

English activity, specificity, and health patterns are built in. To add or
customize pattern-based markers for another language, pass `text_patterns`
keyed by the detected Whisper language code. Pattern matching is
case-insensitive; provide regular expressions for the categories that should
be measured. `activities` accepts `food`, `housework`, `maintenance`,
`errands`, `appointments`, `outings`, `social`, `physical`, and `cognitive`.
`health` accepts `difficulty`, `memory`, `physical_limit`, `assistance`, and
`compensation`. `temporal` and `detail` patterns are used for specificity.
Unspecified categories remain missing (`NaN`), rather than being inferred.
For example:

```python
patterns = {
    "es": {
        "activities": {
            "food": r"\bcocinar\b",
            "housework": r"\blimpiar\b",
            "maintenance": r"\breparar\b",
            "errands": r"\bcompras\b",
            "appointments": r"\bcita médica\b",
            "outings": r"\bpaseo\b",
            "social": r"\bfamilia\b",
            "physical": r"\bcaminar\b",
            "cognitive": r"\bleer\b",
        },
        "health": {
            "difficulty": r"\bno pude\b",
            "memory": r"\bolvidé\b",
            "physical_limit": r"\bcansado\b",
            "assistance": r"\bnecesité ayuda\b",
            "compensation": r"\brecordatorio\b",
        },
        "temporal": r"\bayer\b",
        "detail": r"\bparque\b",
    }
}
audio_result = compute_audio_markers(
    participant,
    include_behavior=True,
    include_health=True,
    text_patterns=patterns,
)
```

Additional syntax languages can be configured with spaCy model names, for
example `syntax_models={"es": "es_core_news_sm", "fr": "your_french_model"}`.
Syntax models are loaded only when a transcript in that language has enough
words for syntax analysis. English syntax uses `en_core_web_sm`; to install it,
run `python -m spacy download en_core_web_sm`. The Spanish model is only checked
when Spanish syntax analysis is needed. TextBlob provides English sentiment;
multilingual sentiment and sentence embeddings are used for other languages.
Missing optional models result in missing values for their corresponding
features. ASR tokenization and lexical and speech-rate measures can vary by
language, so compare those measurements within a language.

### Optional exertion arguments

Exertion defaults are suitable for typical use. The keyword-only arguments let
you customize windowing, signal filtering, and HR availability:

| Argument | Default | Purpose |
|---|---:|---|
| `time_window` | `None` | Pandas frequency string for aggregation windows; when unset, uses `window_seconds`. Keep `window_seconds` consistent when changing this frequency because it is also used in wear-time calculations. |
| `window_seconds` | `30` | Default aggregation window in seconds and the window duration used for wear-time calculations. |
| `hr_weight`, `motion_weight` | `0.3`, `0.7` | Weights for the HR and motion components of combined exertion. |
| `min_hr`, `max_hr` | `35`, `220` | Plausible HR bounds used for peak-HR reporting and HR cleaning when enabled. |
| `motion_robust_z_max`, `hr_robust_z_max` | `6.0`, `4.0` | Robust outlier cutoffs for motion and HR when signal filtering is enabled. |
| `min_motion_samples_per_window`, `min_hr_samples_per_window` | `5`, `1` | Minimum samples required to calculate each window's motion or HR measure. |
| `min_hr_values_for_markers` | `21` | Minimum valid HR samples needed to report HR-derived markers. |
| `filter_signals` | `False` | Enable robust outlier filtering for motion and HR values. |
| `use_short_smoothing` | `None` | Enable/disable rolling-median smoothing. `None` follows `filter_signals`; an explicit boolean overrides it. |
| `motion_smooth_seconds`, `hr_smooth_seconds` | `5`, `15` | Rolling-median window widths in seconds when smoothing is enabled. |

For example:

```python
exertion = compute_exertion_markers(
    participant,
    window_seconds=30,
    filter_signals=True,
    min_hr_values_for_markers=30,
)
```

### Optional gait arguments

Gait accepts keyword-only parameters for sampling rate and bout detection. The
defaults are tuned for 10 Hz wrist motion; adjust them to match the sampling
rate and study protocol:

| Argument | Default | Purpose |
|---|---:|---|
| `fs` | `10.0` | Motion sampling frequency in Hz. |
| `wear_threshold` | `0.5` | Cutoff applied to the binary wear mask; the default retains samples marked worn. |
| `max_gap_factor` | `2.5` | Gap multiplier used to split discontinuous motion segments. |
| `min_steps` | `8` | Minimum steps for an accepted walking bout. |
| `min_bout_seconds` | `4.0` | Minimum walking-bout duration. |
| `max_step_gap_seconds` | `1.5` | Maximum allowed gap between detected steps. |
| `min_cadence`, `max_cadence` | `45.0`, `180.0` | Accepted cadence range in steps per minute. |
| `max_interval_cv` | `0.4` | Maximum coefficient of variation for step intervals. |
| `prominence_scale` | `0.5` | Scale factor used for step-peak prominence detection. |

### Optional activity arguments

Activity markers use the distinct non-missing labels found in the participant's
data. Missing or unlabeled samples are reported as `unknown` coverage and are
excluded from activity-state fractions.

| Argument | Default | Purpose |
|---|---:|---|
| `include_hourly` | `False` | Add a per-state fraction for each clock hour to the daily table. |


## Reading results

`MarkerResult` has `markers`, `daily`, `details`, and `metadata` attributes:

```python
print(exertion.markers.head())  # participant-level summary
print(exertion.daily.head())    # one row per day, when provided by the marker

print(gait.daily[["date", "num_steps", "num_walking_bouts"]])
print(location_result.details["home_locations"])
print(routine.details["state_balance"])
print(activity.daily.head())

for name, table in routine.details.items():
    print(name, table.shape)
```

`markers` contains participant-level values; `daily` contains daily
measurements when that marker provides them. Routine does not provide a `daily`
table: its five-minute records and summaries are in `details`. Use
`details.keys()` to see which additional tables are available, and inspect
`metadata` for marker-specific information such as participant ID and sampling
rate.

The tables are marker-specific. Exertion daily rows include motion-only and
HR-plus-motion exertion, peak HR, HR/motion relationships, and wear time;
participant summaries report statistics across daily measurements. Gait daily
rows include step and bout counts, walking duration, cadence, step-interval and
signal measures; participant summaries include means, standard deviations,
trends, and valid/recorded day counts. In summary column names, `mean_` denotes
the average across days, `std_` the spread, `slope_` the linear trend over time,
and `ac1_` lag-one autocorrelation when consecutive-day data is available;
exact calculations vary by marker. Location returns daily and
participant-level movement summaries, with `details["home_locations"]` and
`details["staypoints"]` for home and staypoint tables. Routine includes
`details["intervals"]`, `details["state_balance"]`, `details["regions"]`,
`details["motion_diagnostics"]`, and `details["location_diagnostics"]`.
Activity daily tables contain observed seconds, unknown-label fraction, state
fractions and bout summaries, transitions, entropy, and time-of-day fractions
and shares. Its participant summaries include means, standard deviations,
calendar-day slopes, and lag-one correlations across daily values.

Missing values are meaningful: a value may be unavailable because the stream
was not supplied or there were too few valid observations. In particular,
exertion's HR-derived metrics are unavailable when fewer than
`min_hr_values_for_markers` valid HR samples exist; motion-only results remain
available. Location can return empty summary tables when there are too few GPS
fixes or no recorded minutes. Check the relevant counts and detail tables
instead of replacing missing values with zero.


### Audio

The audio `MarkerResult.markers` table contains participant-level marker
summaries. `details["recordings"]` contains per-recording measurements and
transcript text. Use `write_audio_transcripts` to save those transcripts; by
default it writes `<original-stem>-transcript.txt`, and `suffix=""` writes
`<original-stem>.txt` instead:

```python
from digital_markers import write_audio_transcripts

write_audio_transcripts(audio_result, "transcripts", suffix="")
```


## Installation

```bash
python -m pip install digital-markers
```

For audio markers:

```bash
python -m pip install "digital-markers[audio]"
```

For development:

```bash
git clone https://github.com/WSU-CASAS/digital-markers.git
cd digital-markers
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```
