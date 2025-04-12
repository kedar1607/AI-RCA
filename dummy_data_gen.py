import csv
import random
from datetime import datetime, timedelta

#######################################
# Helper Functions and Bug Window Generation
#######################################

def generate_bug_windows(num_windows, min_duration, max_duration, period_start, period_end):
    """Generate a list of bug windows (start, end) within the given period."""
    windows = []
    total_seconds = int((period_end - period_start).total_seconds())
    for _ in range(num_windows):
        start_offset = random.randint(0, total_seconds)
        win_start = period_start + timedelta(seconds=start_offset)
        duration = random.randint(min_duration, max_duration)
        win_end = win_start + timedelta(seconds=duration)
        if win_end > period_end:
            win_end = period_end
        windows.append((win_start, win_end))
    return windows

def format_window(win):
    """Format a bug window tuple as a string."""
    return f"{win[0].strftime('%Y-%m-%d %H:%M:%S')} - {win[1].strftime('%Y-%m-%d %H:%M:%S')}"

def random_date_range(start, end):
    """Return a random datetime between start and end."""
    delta = end - start
    random_seconds = random.randrange(int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)

#######################################
# Simulation Time Ranges
#######################################
# January (training) period:
def start():
    training_start = datetime(2025, 1, 1, 8, 0, 0)
    training_end = datetime(2025, 1, 31, 23, 59, 59)

    # February (CV) period:
    cv_start = datetime(2025, 2, 1, 0, 0, 0)
    cv_end   = datetime(2025, 2, 5, 23, 59, 59)

    #######################################
    # Pre-generate Bug Windows for January
    #######################################
    # vendor outages: 4 windows (10-30 minutes)
    vendor_outage_windows_count = 16
    vendor_outage_window_min_duration = 600
    vendor_outage_window_max_duration = 1800
    vendor_outage_windows = generate_bug_windows(vendor_outage_windows_count, vendor_outage_window_min_duration, vendor_outage_window_max_duration, training_start, training_end)
    # Video outages: 8 windows (15-29 minutes)
    video_outage_windows_count = 32
    video_outage_window_min_duration = 600
    video_outage_window_max_duration = 1800
    video_windows = generate_bug_windows(video_outage_windows_count, video_outage_window_min_duration, video_outage_window_max_duration, training_start, training_end)
    # Audio outages: 7 windows (1-3 hours)
    audio_outage_windows_count = 28
    audio_outage_window_min_duration = 3600
    audio_outage_window_max_duration = 10800
    audio_windows = generate_bug_windows(audio_outage_windows_count, audio_outage_window_min_duration, audio_outage_window_max_duration, training_start, training_end)
    # Twitter login outage: 1 window of 7 hours
    external_login_outage_windows_count = 8
    external_login_outage_window_min_duration = 3600
    external_login_outage_window_max_duration = 10800
    external_login_window = generate_bug_windows(external_login_outage_windows_count, external_login_outage_window_min_duration, external_login_outage_window_max_duration, training_start, training_end)

    #######################################
    # Pre-define Bug Windows for February (CV)
    #######################################
    cv_vendor_outage_window = (datetime(2025, 2, 1, 2, 0, 0), datetime(2025, 2, 1, 6, 0, 0))
    cv_video_outage_window = (datetime(2025, 2, 1, 8, 0, 0), datetime(2025, 2, 1, 8, 50, 0))
    cv_login_outage_window = (datetime(2025, 2, 1, 10, 0, 0), datetime(2025, 2, 1, 16, 0, 0))
    cv_audio_outage_window = (datetime(2025, 2, 1, 12, 0, 0), datetime(2025, 2, 1, 16, 0, 0))

    #######################################
    # Event Definitions for Android and iOS
    #######################################
    # For DD and AS we include the corresponding screen class names.
    event_definitions_android = {
        "Splash": {"dd": ["SplashScreenActivity displayed"], "as": ["SplashScreenActivity impression recorded"], "be": []},
        "LoginPage": {"dd": ["LoginActivity displayed"], "as": ["LoginActivity impression recorded"], "be": []},
        "LoginAttempt": {"dd": ["User submitted credentials in LoginActivity"], "as": ["Login button clicked in LoginActivity"], "be": ["Authentication request received", "Credentials verification started"]},
        "LoginResultSuccess": {"dd": ["LoginActivity: Login successful"], "as": ["Login success recorded from LoginActivity"], "be": ["User authenticated", "Session token generated"]},
        "LoginResultFailure": {"dd": ["LoginActivity: Login failed"], "as": ["Login failure recorded from LoginActivity"], "be": ["Authentication error", "Invalid credentials"]},
        "Home": {"dd": ["HomeScreenActivity displayed"], "as": ["HomeScreenActivity impression recorded"], "be": ["Home content requested", "Fetching home articles", "Home service responded"]},
        "ArticleText": {"dd": ["ArticleTextActivity opened"], "as": ["ArticleTextActivity impression recorded"], "be": ["Text article details fetched", "Text content loaded"]},
        "ArticleAudio": {"dd": ["ArticleAudioActivity opened"], "as": ["ArticleAudioActivity impression recorded"], "be": ["Audio content fetched", "Streaming audio started"]},
        "ArticleVideo": {"dd": ["ArticleVideoActivity opened"], "as": ["ArticleVideoActivity impression recorded"], "be": ["Video content fetched", "Streaming video delivered"]},
        "ArticleError": {"dd": ["ArticleErrorActivity: Failed to load article"], "as": ["ArticleErrorActivity impression recorded"], "be": ["Article fetch error", "Fallback to cached article"]},
        "VideoArticleError": {"dd": ["VideoViewActivity: Failed to load the video"], "as": ["VideoViewActivity error impression recorded"], "be": ["Failed to deliver the frames for the video"]},
        "AudioArticleError": {"dd": ["AudioViewActivity: Failed to load the audio"], "as": ["AudioViewActivityr error impression recorded"], "be": ["Failed to deliver the audio"]},
        "Search": {"dd": ["SearchActivity displayed"], "as": ["SearchActivity impression recorded"], "be": ["Search request processed", "Search results returned"]},
        "Share": {"dd": ["ShareActivity: User clicked share"], "as": ["Share event recorded from ShareActivity"], "be": ["Share request processed"]},
        "Exit": {"dd": ["User exited app from LastScreenActivity"], "as": ["Exit event recorded"], "be": []}
    }

    event_definitions_ios = {
        "Splash": {"dd": ["SplashScreenViewController displayed"], "as": ["SplashScreenViewController impression recorded"], "be": []},
        "LoginPage": {"dd": ["LoginViewController displayed"], "as": ["LoginViewController impression recorded"], "be": []},
        "LoginAttempt": {"dd": ["User submitted credentials in LoginViewController"], "as": ["Login button clicked in LoginViewController"], "be": ["Authentication request received", "Credentials verification started"]},
        "LoginResultSuccess": {"dd": ["LoginViewController: Login successful"], "as": ["Login success recorded from LoginViewController"], "be": ["User authenticated", "Session token generated"]},
        "LoginResultFailure": {"dd": ["LoginViewController: Login failed"], "as": ["Login failure recorded from LoginViewController"], "be": ["Authentication error", "Invalid credentials"]},
        "Home": {"dd": ["HomeScreenViewController displayed"], "as": ["HomeScreenViewController impression recorded"], "be": ["Home content requested", "Fetching home articles", "Home service responded"]},
        "ArticleText": {"dd": ["ArticleTextViewController opened"], "as": ["ArticleTextViewController impression recorded"], "be": ["Text article details fetched", "Text content loaded"]},
        "ArticleAudio": {"dd": ["ArticleAudioViewController opened"], "as": ["ArticleAudioViewController impression recorded"], "be": ["Audio content fetched", "Streaming audio started"]},
        "ArticleVideo": {"dd": ["ArticleVideoViewController opened"], "as": ["ArticleVideoViewController impression recorded"], "be": ["Video content fetched", "Streaming video delivered"]},
        "ArticleError": {"dd": ["ArticleErrorViewController: Failed to load article"], "as": ["ArticleErrorViewController impression recorded"], "be": ["Article fetch error", "Fallback to cached article"]},
        "VideoArticleError": {"dd": ["VideoViewController: Failed to load the video"], "as": ["VideoViewController error impression recorded"], "be": ["Failed to deliver the frames for the video"]},
        "AudioArticleError": {"dd": ["AudioViewController: Failed to load the audio"], "as": ["AudioViewController error impression recorded"], "be": ["Failed to deliver the audio"]},
        "Search": {"dd": ["SearchViewController displayed"], "as": ["SearchViewController impression recorded"], "be": ["Search request processed", "Search results returned"]},
        "Share": {"dd": ["ShareViewController: User clicked share"], "as": ["Share event recorded from ShareViewController"], "be": ["Share request processed"]},
        "Exit": {"dd": ["User exited app from LastScreenViewController"], "as": ["Exit event recorded"], "be": []}
    }

    #######################################
    # Global Lists to Hold Logs
    #######################################
    training_dd_logs, training_be_logs, training_as_logs = [], [], []
    cv_dd_logs, cv_be_logs, cv_as_logs = [], [], []

    # We'll use these for simulation sessions.
    article_sources = ["wapo", "nyt", "ap", "bbc", "cnn"]

    #######################################
    # Utility Function to Add a Log Entry
    #######################################
    def add_log(log_list, system, timestamp, session_id, event_idx, event_name, message, platform=None, event_type=None, article_id="", corr_id=""):
        entry = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "event_idx": event_idx,
            "event_name": event_name,
            "log_message": message
        }
        if platform:
            entry["platform"] = platform
        if event_type:
            entry["event_type"] = event_type
        if article_id:
            entry["article_id"] = article_id
        if corr_id:
            entry["correlation_id"] = corr_id
        log_list.append(entry)

    #######################################
    # Simulation Function for One Session
    #######################################
    def simulate_session(is_training, session_id, session_start, event_defs, bug_windows):
        # For login events, choose a login method.
        def choose_login_method():
            return random.choices(["Standard", "Twitter", "Google"], weights=[20,40,40], k=1)[0]
        
        events = []
        events.append({"event": "Splash"})
        login_success = False
        login_method = "Guest"
        if random.random() < 0.8:
            events.append({"event": "LoginPage"})
            login_method = choose_login_method()
            events.append({"event": "LoginAttempt", "login_method": login_method})
            login_success = (random.random() < 0.95)
            # For Twitter outage bug in Jan:
            if login_method == "Twitter" and is_training:
                for (ws, we) in bug_windows.get("login_outage", []):
                    if ws and ws <= session_start <= we:
                        login_success = False
            events.append({"event": "LoginResultSuccess" if login_success else "LoginResultFailure", "login_method": login_method})
            if not login_success:
                events.append({"event": "Exit"})
        else:
            # reinforce guest login
            login_method = "Guest"
        
        if (login_success or login_method == "Guest"):
            events.append({"event": "Home"})
            num_articles = random.randint(0, 5)
            for _ in range(num_articles):
                art_event = random.choices(["ArticleText", "ArticleAudio", "ArticleVideo"], weights=[40,30,30], k=1)[0]
                events.append({"event": art_event})
                if random.random() < 0.3:
                    events.append({"event": "Share"})
            if random.random() < 0.25:
                events.append({"event": "Search"})
            events.append({"event": "Exit"})
        
        current_time = session_start
        event_counter = 0
        for evt in events:
            event_counter += 1
            current_time += timedelta(seconds=random.randint(5, 60))
            ev_name = evt["event"]
            bug_override = None  # to hold bug type if injected
            src = random.choice(article_sources) # get the source first before doing anything else
            
            # --- January Bug Injections ---
            if is_training and current_time <= training_end:
                # AP outage: for Article events with source "ap" (all types of articles failed)
                if ev_name == "ArticleText":
                    for (ws, we) in bug_windows.get("vendor_outage", []):
                        if ws <= current_time <= we:
                            ev_name = "ArticleError"
                            bug_override = "vendor_outage"
                            break
        
                # Video outage: if event is ArticleVideo, with 50% chance
                if ev_name == "ArticleVideo":
                    for (ws, we) in bug_windows.get("Video", []):
                        if ws <= current_time <= we and random.random() < 0.5:
                            ev_name = "VideoArticleError"
                            bug_override = "Video"
                            break
                        
                # Audio outage: if event is ArticleAudio, force failure
                if ev_name == "ArticleAudio":
                    for (ws, we) in bug_windows.get("Audio", []):
                        if ws <= current_time <= we:
                            ev_name = "AudioArticleError"
                            bug_override = "Audio"
                            break
                        
                # Twitter login outage: force failure if login_method is Twitter
                if login_method == "Twitter":
                    for (ws, we) in bug_windows.get("login_outage", []):
                        if ws and ws <= current_time <= we:
                            if ev_name == "LoginResultSuccess":
                                ev_name = "LoginResultFailure"
                                bug_override = "login_outage"
                                break
            
            # --- February Bug Injections ---
            if (not is_training) and (cv_start <= current_time <= cv_end):
                if ev_name in ["ArticleText", "ArticleAudio", "ArticleVideo"]:
                    if src == "cnn":
                        cnn_start, cnn_end = bug_windows.get("CNN", (None, None))
                        if cnn_start and cnn_start <= current_time <= cnn_end:
                            ev_name = "ArticleError"
                            bug_override = "CNN"
                            break

                if ev_name == "ArticleAudio":
                    ae_start, ae_end = bug_windows.get("AudioFeb", (None, None))
                    if ae_start <= current_time <= ae_end:
                        ev_name = "AudioArticleError"
                        bug_override = "AudioFeb"
                        break
                
                if ev_name == "ArticleVideo":
                    fv_start, fv_end = bug_windows.get("VideoFeb", (None, None))
                    if fv_start and fv_start <= current_time <= fv_end:
                        ev_name = "VideoArticleError"
                        bug_override = "VideoFeb"
                        break
                if (platform == "Android") and ("login_method" in evt) and (evt["login_method"]=="Google"):
                    g_start, g_end = bug_windows.get("Google", (None, None))
                    if g_start and g_start <= current_time <= g_end:
                        if ev_name == "LoginResultSuccess":
                            ev_name = "LoginResultFailure"
                            bug_override = "Google"
                            break
            
            final_event = ev_name
            
            # Generate correlation ID if event has BE call.
            if event_defs.get(final_event, {}).get("be"):
                corr_id = f"CORR-{session_id}-{event_counter}"
            else:
                corr_id = ""
            
            # For article events, assign an article ID.
            article_id = ""
            if final_event in ["ArticleText", "ArticleAudio", "ArticleVideo", "ArticleError", "VideoArticleErro", "AudioArticleError"]:
                art_src = src
                article_id = f"{art_src}{random.randint(10000000, 99999999)}"
            
            # Set default log messages.
            dd_msg = event_defs.get(final_event, {}).get("dd", [""])[0]
            as_msg = event_defs.get(final_event, {}).get("as", [""])[0]
            if event_defs.get(final_event, {}).get("be"):
                be_msg = random.choice(event_defs.get(final_event, {}).get("be", [""]))
            else:
                be_msg = ""
            
            # If a bug was injected, override messages accordingly.
            if bug_override:
                if bug_override == "vendor_outage":
                    dd_msg = f"ArticleErrorScreen: {article_id} from {src} Article failed to load"
                    as_msg = "ArticleErrorScreen: Article Error Screen impression recorded"
                    be_msg = f"External API for {src} articles not responding; https://{src}.com/v1/headlines/{article_id} returned 400"
                elif bug_override == "Video":
                    dd_msg = f"VideoScreenError: Video Streaming Network Error from {src} for {article_id}"
                    as_msg = "VideoScreenError: Video Streaming Error impression recorded"
                    be_msg = f"Video streaming library overloaded; failed to serve video from {src} https://mynewsclips.stream/{article_id}"
                elif bug_override == "Audio":
                    dd_msg = "AudioScreenError: Audio Streaming API throwing an error"
                    as_msg = "AudioScreenError: Audio Error Screen impression recorded"
                    be_msg = f"AI Text-to-Speech service down; audio article failed https://mynewsaudiostream.stream/{article_id} has returned Internal Server Error (500) due to TTS service outage"
                elif bug_override == "login_outage":
                    dd_msg = f"LoginActivity: User login using {login_method} failed because of Server Error"
                    as_msg = "Login failure recorded: Login Failure message displayed"
                    be_msg = f"{login_method} authentication error: Exceeded Twitter API rate limit"
                elif bug_override == "CNN":
                    dd_msg = f"ArticleErrorScreen: {article_id} from {src} Article failed to load"
                    as_msg = "ArticleErrorScreen: Article Error Screen impression recorded"
                    be_msg = f"External API for {src} articles not responding; https://{src}.com/v1/headlines/{article_id} returned 400"
                elif bug_override == "VideoFeb":
                    dd_msg = f"VideoScreenError: Video Streaming Network Error from {src} for {article_id}"
                    as_msg = "VideoScreenError: Video Streaming Error impression recorded"
                    be_msg = f"Video streaming library overloaded; failed to serve video from {src} https://mynewsclips.stream/{article_id}"
                elif bug_override == "Google":
                    dd_msg = f"LoginActivity: User login using {login_method} failed because of Server Error"
                    as_msg = "Login failure recorded: Login Failure message displayed"
                    be_msg = f"{login_method} authentication error: Client SDK is deprecated, please upgrade to the version >= 6.7.8"
                elif bug_override == "AudioFeb":
                    dd_msg = "AudioScreenError: Audio Streaming API throwing an error"
                    as_msg = "AudioScreenError: Audio Error Screen impression recorded"
                    be_msg = f"AI Text-to-Speech service down; audio article failed https://mynewsaudiostream.stream/{article_id} has returned Internal Server Error (500) due to TTS service outage"
                # if event_defs.get(final_event, {}).get("be"):
                #     num_lines = random.randint(1, 3)
                #     be_lines = []
                #     for i in range(num_lines):
                #         line_time = (current_time + timedelta(seconds=i * 3)).strftime("%Y-%m-%d %H:%M:%S")
                #         be_lines.append(f"{line_time} | {be_msg}")
                #     be_msg = "\n".join(be_lines)
            
            # Add logs to the corresponding global lists.
            add_log(current_dd_logs, "DD", current_time, session_id, event_counter, final_event, dd_msg, platform=platform, article_id=article_id, corr_id=corr_id)
            as_event_type = "click" if final_event in ["LoginAttempt", "Share"] else "impression"
            add_log(current_as_logs, "AS", current_time, session_id, event_counter, final_event, as_msg, event_type=as_event_type, article_id=article_id)
            if event_defs.get(final_event, {}).get("be"):
                add_log(current_be_logs, "BE", current_time, session_id, event_counter, final_event, be_msg, article_id=article_id, corr_id=corr_id)

    #######################################
    # Simulate Sessions for January and February
    #######################################
    NUM_SESSIONS_FOR_TRAINING = 20000  # January sessions
    NUM_SESSIONS_CROSS_VALIDATION = 5000   # February sessions

    # Simulate January sessions
    current_dd_logs = training_dd_logs
    current_be_logs = training_be_logs
    current_as_logs = training_as_logs
    for sess in range(1, NUM_SESSIONS_FOR_TRAINING + 1):
        session_start = random_date_range(training_start, training_end)
        platform = random.choice(["Android", "iOS"])
        sess_prefix = "sess-A" if platform == "Android" else "sess-I"
        session_id = f"{sess_prefix}{10000 + sess}"
        event_defs = event_definitions_android if platform == "Android" else event_definitions_ios
        simulate_session(True, session_id, session_start, event_defs, {
            "vendor_outage": vendor_outage_windows,
            "Video": video_windows,
            "Audio": audio_windows,
            "login_outage": external_login_window
        })

    # Simulate February sessions
    current_dd_logs = cv_dd_logs
    current_be_logs = cv_be_logs
    current_as_logs = cv_as_logs
    for sess in range(1, NUM_SESSIONS_CROSS_VALIDATION + 1):
        session_start = random_date_range(cv_start, cv_end)
        platform = random.choice(["Android", "iOS"])
        sess_prefix = "sess-A" if platform == "Android" else "sess-I"
        session_id = f"{sess_prefix}{10000 + sess}"
        event_defs = event_definitions_android if platform == "Android" else event_definitions_ios
        cv_bug_windows = {
            "CNN": cv_vendor_outage_window,
            "VideoFeb": cv_video_outage_window,
            "Google": cv_login_outage_window,
            "AudioFeb": cv_audio_outage_window
        }
        simulate_session(False, session_id, session_start, event_defs, cv_bug_windows)

    #######################################
    # Write Combined Log CSV Files
    #######################################
    all_dd_logs = training_dd_logs + cv_dd_logs
    all_be_logs = training_be_logs + cv_be_logs
    all_as_logs = training_as_logs + cv_as_logs

    with open("datadog_mobile_logs.csv", "w", newline="") as f:
        fieldnames = ["timestamp", "session_id", "event_idx", "event_name", "platform", "log_message", "article_id", "correlation_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_dd_logs)

    with open("backend_logs.csv", "w", newline="") as f:
        fieldnames = ["timestamp", "session_id", "event_idx", "event_name", "correlation_id", "service", "log_message", "article_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_be_logs)

    with open("analytics_logs.csv", "w", newline="") as f:
        fieldnames = ["timestamp", "session_id", "event_idx", "event_name", "event_type", "log_message", "article_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_as_logs)

    print("Generated log CSV files:")
    print(f"  - DataDog logs: {len(all_dd_logs)} rows")
    print(f"  - Backend logs: {len(all_be_logs)} rows")
    print(f"  - Analytics logs: {len(all_as_logs)} rows")

    #######################################
    # Generate Jira Tickets CSV Files
    #######################################
    # Training Tickets (January bugs) – include RCA and Time_Window
    training_tickets = [
        {
            "Issue_ID": "JIRA-TRAIN-001",
            "Project": "Live Issues",
            "Summary": "AP Articles not loading",
            "Description": "Users reported that AP articles were not loading in this time window",
            "RCA": "Article vendor outage",
            "Time_Window": [format_window(i) for i in vendor_outage_windows]
        },
        {
            "Issue_ID": "JIRA-TRAIN-002",
            "Project": "Live Issues",
            "Summary": "NFL Video Streaming Failure",
            "Description": "Too many video streaming failure events in this time window",
            "RCA": "Video Service Overloaded. Possibly Peak Time Issue",
            "Time_Window": [format_window(i) for i in video_windows]
        },
        {
            "Issue_ID": "JIRA-TRAIN-003",
            "Project": "Live Issues",
            "Summary": "NYT Audio Articles Error",
            "Description": "Audio articles reported errors in this time window.",
            "RCA": "Audio Service not returning soundbytes. Probably TTS Service Down",
            "Time_Window": [format_window(i) for i in audio_windows]
        },
        {
            "Issue_ID": "JIRA-TRAIN-004",
            "Project": "Live Issues",
            "Summary": "Twitter Login Failure",
            "Description": "Sessions using Twitter login encountered failures for a prolonged period.",
            "RCA": "Login vendor is throwing an error. Please check backend logs for more details",
            "Time_Window": [format_window(i) for i in external_login_window]
        }
    ]
    for ticket in training_tickets:
        created_date = random_date_range(datetime(2025, 2, 1), datetime(2025, 2, 2))
        ticket["Created_Date"] = created_date.strftime("%Y-%m-%d %H:%M:%S")

    # CV Tickets (February bugs) – RCA omitted
    cv_tickets = [
        {
            "Issue_ID": "JIRA-CV-001",
            "Project": "Live Issues",
            "Summary": "CNN Articles not loading",
            "Description": "Users reported that CNN articles were not loading in this time window",
            "Time_Window": format_window(cv_vendor_outage_window)
        },
        {
            "Issue_ID": "JIRA-CV-002",
            "Project": "Live Issues",
            "Summary": "SOTU Video Streaming Failure",
            "Description": "Too many video streaming failure events in this time window",
            "Time_Window": format_window(cv_video_outage_window)
        },
        {
            "Issue_ID": "JIRA-CV-003",
            "Project": "Live Issues",
            "Summary": "Google Login Failure",
            "Description": "Android Sessions using Google login encountered failures for a prolonged period.",
            "Time_Window": format_window(cv_login_outage_window)
        },
        {
            "Issue_ID": "JIRA-CV-004",
            "Project": "Live Issues",
            "Summary": "Wapo Audio Articles Error",
            "Description": "Audio articles reported errors in this time window.",
            "Time_Window": format_window(cv_audio_outage_window)
        }
    ]
    for ticket in cv_tickets:
        created_date = random_date_range(datetime(2025, 2, 6), datetime(2025, 2, 7))
        ticket["Created_Date"] = created_date.strftime("%Y-%m-%d %H:%M:%S")

    with open("jira_tickets_training.csv", "w", newline="") as f:
        fieldnames = ["Issue_ID", "Project", "Summary", "Description", "Created_Date", "Time_Window", "RCA"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_tickets)

    with open("jira_tickets_cv.csv", "w", newline="") as f:
        fieldnames = ["Issue_ID", "Project", "Summary", "Description", "Created_Date", "Time_Window"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cv_tickets)

    print("Generated Jira tickets CSV files for Training and CV data.")

if __name__ == "__main__":
    start()