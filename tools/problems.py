"""
Latent problem catalogue -- v2, compositional.

v1 produced perfectly separable classes: every problem owned distinctive content
words, so classification hit 100% and the study had nothing to measure. Real help
desk corpora are not separable, for three reasons that v2 reproduces explicitly.

1. SHARED SYMPTOM FAMILIES. Users open with the same handful of complaints --
   "I cannot login", "access denied", "it is very slow", "it is not working" --
   regardless of which system is actually at fault. These pools are shared
   verbatim across problems in DIFFERENT categories. This is the phenomenon in
   Sec. 1.2 ("the same terms may appear in tickets that in fact relate to
   unrelated problem categories") and the limitation of keyword matching in
   Sec. 2.3.

2. OPTIONAL DISAMBIGUATING CONTEXT. Each problem also carries `contexts` -- the
   detail that actually identifies the system. Real users supply it only
   sometimes. generate_dataset.py includes it with probability P_CONTEXT; when
   omitted the ticket is GENUINELY ambiguous and no model can resolve it above
   the class prior. This sets a real, honest ceiling on achievable accuracy.

3. COMPOSITIONAL SURFACE FORM. Text is assembled from slots (greeting, core,
   context, location, time, request) rather than drawn from a small fixed pool,
   so near-duplicate tickets are rare and models must generalise.

`cores` may be generic; `contexts` carries the signal; `resolution` is the
ground-truth resolution note for Objective 4.
"""

CATEGORIES = [
    "Network & Connectivity",
    "Account & Access Management",
    "Email & Collaboration",
    "Learning Management System",
    "Student Information System",
    "Application Support",
    "Hardware & Equipment",
]

TEAMS = [
    "Network Team", "Systems & Accounts Team", "eLearning Support Unit",
    "MIS / SIS Support Team", "Applications Support Team",
    "Hardware & Maintenance Team", "Help Desk (Tier 1)",
]

# ---------------------------------------------------------------------------
# Shared symptom pools -- deliberately reused across unrelated categories.
# ---------------------------------------------------------------------------
LOGIN = [
    "i cannot login", "i am unable to log in", "login is not working",
    "the system is not accepting my login", "i keep getting invalid credentials",
    "my login keeps failing", "it refuses to let me in",
    "i am unable to sign in", "it says my details are wrong",
]
ACCESS = [
    "i cannot access the system", "access is denied", "i have no access",
    "it says access denied", "i am unable to get in", "i am locked out",
    "permission is denied when i try", "i cannot open it",
]
SLOW = [
    "it is extremely slow", "everything is very slow", "it takes forever to load",
    "performance is very poor", "it keeps hanging", "it is slow and keeps timing out",
    "it is dragging badly", "response is very slow",
]
BROKEN = [
    "it is not working", "it has stopped working", "it is down",
    "it has refused to work", "nothing is happening", "it is not functioning",
    "it keeps failing", "it is giving an error",
]
MISSING = [
    "it is not showing", "it is not appearing", "i cannot see it",
    "it is missing", "it does not appear anywhere", "it has disappeared",
]

GENERIC_SUBJECTS = [
    "Urgent assistance required", "Help needed", "ICT issue",
    "Request for support", "Problem with system", "Assistance required",
    "Issue affecting my work", "Kindly assist", "Support request",
]

P = []


def prob(pid, cat, sub, team, weight, subjects, cores, contexts, resolution):
    P.append(dict(pid=pid, category=cat, subcategory=sub, team=team, weight=weight,
                  subjects=subjects, cores=cores, contexts=contexts,
                  resolution=resolution))


# ------------------------------------------------------------------ NETWORK
_ = prob("NET-WIFI-NOCONN", "Network & Connectivity", "Wi-Fi", "Network Team", 88,
     ["No wifi connection", "Cannot connect to wifi", "Wireless not working",
      "WiFi problem", "Unable to join the wireless network"],
     BROKEN + ACCESS + ["i cannot connect", "the connection keeps dropping",
                        "it will not connect at all"],
     ["this is on the wireless network", "my laptop will not join mu-wifi",
      "the wifi shows but does not connect", "it keeps asking for the wifi password",
      "the wireless access point seems to be off", "there is no wifi in our block"],
     "Reset the wireless access point serving the block and re-authenticated the user device on MU-WiFi. Confirmed connectivity restored.")

_ = prob("NET-SLOW", "Network & Connectivity", "Performance", "Network Team", 54,
     ["Internet is slow", "Slow connection", "Poor network performance",
      "Network very slow", "Bandwidth problem"],
     SLOW + BROKEN,
     ["the internet connection is the problem", "browsing is very slow",
      "downloads crawl even on a good signal", "the whole department's link is affected",
      "it is worse in the afternoon when everyone is online"],
     "Investigated bandwidth utilisation on the department switch, identified a broadcast storm from a faulty device, isolated the port and normalised throughput.")

_ = prob("NET-LAN-PORT", "Network & Connectivity", "LAN", "Network Team", 34,
     ["Network point not working", "LAN port dead", "No network on desktop",
      "Cable not detecting", "Wall port problem"],
     BROKEN + ["there is no connection at all", "it does not detect anything"],
     ["the network point on the wall is dead", "the cable shows unplugged",
      "the ethernet port is not giving a link", "i have tried another network cable"],
     "Traced the wall port to the patch panel, re-terminated the faulty jack and re-patched to an active switch port. Link confirmed at 1Gbps.")

_ = prob("NET-VPN", "Network & Connectivity", "VPN / Remote access", "Network Team", 21,
     ["VPN not connecting", "Remote access problem", "Cannot work from home",
      "VPN authentication failing"],
     LOGIN + BROKEN + ACCESS,
     ["this is on the vpn client", "i am trying to connect remotely from home",
      "the vpn says connection timed out", "i need remote access to the office systems"],
     "Re-issued the VPN profile, confirmed the account was in the remote-access group and verified successful tunnel establishment from an external network.")

# ---------------------------------------------------------- ACCOUNT / ACCESS
_ = prob("ACC-PWD-RESET", "Account & Access Management", "Password reset", "Systems & Accounts Team", 118,
     ["Password reset", "Forgot password", "Password expired",
      "Kindly reset my password", "Cannot login"],
     LOGIN + ACCESS,
     ["i have forgotten my password", "my password has expired and needs resetting",
      "i need my password reset", "the password i changed is not being accepted",
      "this is for my staff domain account"],
     "Verified the requester's identity against staff/student records, reset the account password in Active Directory and issued a temporary password with a forced change at next logon.")

_ = prob("ACC-LOCKED", "Account & Access Management", "Account lockout", "Systems & Accounts Team", 47,
     ["Account locked", "Locked out", "Account disabled", "Cannot login account locked"],
     LOGIN + ACCESS,
     ["my account has been locked after several attempts",
      "the system says my account is disabled", "i need my account unlocked",
      "it locked itself after i typed the wrong password too many times"],
     "Confirmed the lockout source, cleared the lockout in Active Directory, unlocked the account and advised the user on clearing cached credentials on secondary devices.")

_ = prob("ACC-NEW", "Account & Access Management", "New account", "Systems & Accounts Team", 39,
     ["New staff account", "Account creation request", "New user setup",
      "Request for new account"],
     ["kindly create an account", "we need an account opened", "i am requesting a new account"],
     ["this is for a newly employed member of staff", "the new tutorial fellow reported this week",
      "we need a domain account and email address created", "please set up access for the new technician"],
     "Created the domain account and mailbox from the approved appointment letter, assigned the departmental security groups and issued credentials to the user.")

_ = prob("ACC-PERMS", "Account & Access Management", "Permissions", "Systems & Accounts Team", 29,
     ["Access denied to shared folder", "Permission request", "Cannot open shared drive",
      "Access rights needed"],
     ACCESS + BROKEN,
     ["this is the departmental shared folder on the server",
      "my colleague can open it but i cannot", "i need rights to the shared drive",
      "it is the shared directory with the semester files"],
     "Added the user to the correct departmental security group after confirming approval from the head of department, and verified access to the shared resource.")

# -------------------------------------------------------------------- EMAIL
_ = prob("EML-SEND-FAIL", "Email & Collaboration", "Mail delivery", "Systems & Accounts Team", 44,
     ["Cannot send email", "Email not going out", "Mail delivery failure",
      "Outlook problem", "Email issue"],
     BROKEN + ["messages are bouncing back", "it fails whenever i press send"],
     ["outgoing mail fails but incoming arrives", "my emails stay in the outbox",
      "i get a delivery failure notice each time", "this is on outlook"],
     "Identified an incorrect SMTP configuration on the client, corrected the outgoing server settings and authentication method, and confirmed successful test delivery.")

_ = prob("EML-MAILBOX-FULL", "Email & Collaboration", "Mailbox quota", "Systems & Accounts Team", 24,
     ["Mailbox full", "Email quota exceeded", "Not receiving emails", "Storage limit reached"],
     BROKEN + MISSING + ["nothing is coming in any more"],
     ["my mailbox has exceeded its storage limit", "i need my mailbox quota increased",
      "no mail is arriving because the box is full"],
     "Advised the user on archiving, purged deleted items and increased the mailbox quota in line with the ICT storage policy. Confirmed mail flow resumed.")

_ = prob("EML-MEETING", "Email & Collaboration", "Online meetings", "Applications Support Team", 19,
     ["Cannot join meeting", "Zoom not working", "Teams problem", "Online meeting issue"],
     BROKEN + ACCESS + ["it closes as soon as it opens"],
     ["this is for an online meeting", "participants cannot join the session i created",
      "the meeting link is not opening", "audio is not detected in the conferencing app"],
     "Reinstalled the conferencing client, granted the required microphone and camera permissions and validated a test meeting session with the user.")

# ---------------------------------------------------------------------- LMS
_ = prob("LMS-LOGIN", "Learning Management System", "Access", "eLearning Support Unit", 58,
     ["Cannot login to eLearning", "eLearning access problem", "LMS login failing",
      "Cannot login", "Portal login issue"],
     LOGIN + ACCESS,
     ["this is on the elearning portal", "i cannot get into the e-learning platform",
      "my email password works but elearning does not accept it",
      "i need to get in to upload my course materials"],
     "Confirmed the account existed on the LMS but was not synchronised with the directory; re-synchronised the user record and verified successful login.")

_ = prob("LMS-COURSE-MISSING", "Learning Management System", "Course enrolment", "eLearning Support Unit", 49,
     ["Course not appearing", "Unit missing on portal", "Cannot see my course",
      "Unit not showing"],
     MISSING + BROKEN,
     ["my units are not on the elearning dashboard", "the course i teach is not on the platform",
      "the course page is not visible to my students",
      "i registered the unit but it is not on e-learning"],
     "Verified enrolment records against the student information system, re-ran the enrolment synchronisation for the affected unit and confirmed the course was visible to the user.")

_ = prob("LMS-UPLOAD", "Learning Management System", "Content upload", "eLearning Support Unit", 31,
     ["Cannot upload notes", "Upload failing", "File will not upload", "Upload error"],
     BROKEN + ["it hangs part way through", "it refuses to accept the file"],
     ["i am uploading lecture notes to the elearning portal",
      "the platform says my video file is too large",
      "the assignment file will not go up to the course page"],
     "Increased the per-file upload limit for the course area and advised the user on compressing large media before upload. Confirmed successful upload.")

_ = prob("LMS-QUIZ", "Learning Management System", "Assessment", "eLearning Support Unit", 27,
     ["Students cannot submit CAT", "Quiz not opening", "Assignment submission problem",
      "Online exam issue"],
     BROKEN + ACCESS + MISSING,
     ["my students cannot submit the cat on the platform",
      "the quiz will not open for the class and the deadline is today",
      "the assignment submission link on the course page is closed"],
     "Corrected the assessment availability window and group restrictions on the activity, extended the deadline as authorised by the lecturer and confirmed submissions were accepted.")

# ---------------------------------------------------------------------- SIS
_ = prob("SIS-REG", "Student Information System", "Unit registration", "MIS / SIS Support Team", 54,
     ["Cannot register units", "Registration not working", "Portal registration problem",
      "Cannot login", "Unable to register"],
     LOGIN + ACCESS + BROKEN,
     ["i am trying to register my units for the semester",
      "this is on the student portal", "the registration page errors after i select units",
      "it says my registration is not allowed"],
     "Confirmed fee-balance and academic-status flags blocking registration, cleared the eligibility block after finance confirmation and verified the student completed unit registration.")

_ = prob("SIS-FEE", "Student Information System", "Fee statement", "MIS / SIS Support Team", 37,
     ["Fee statement problem", "Payment not reflecting", "Fees not updated",
      "Balance still showing"],
     MISSING + BROKEN,
     ["i paid my fees but it is not reflecting in the portal",
      "my fee statement does not show the amount paid",
      "the portal still shows a balance after i cleared", "i have attached the bank slip"],
     "Reconciled the bank transaction reference against the finance ledger, posted the missing receipt to the student account and confirmed the updated statement in the portal.")

_ = prob("SIS-TRANSCRIPT", "Student Information System", "Results / transcripts", "MIS / SIS Support Team", 29,
     ["Missing marks", "Results not showing", "Transcript incomplete", "Cannot view results"],
     MISSING + ACCESS + BROKEN,
     ["my results for two units are not in the portal",
      "the transcript is missing marks from last semester",
      "the results page is blank after i log in"],
     "Traced the unresolved marks to an unapproved mark sheet, escalated to the school examinations office for approval and confirmed the results displayed after posting.")

_ = prob("SIS-PORTAL-SLOW", "Student Information System", "Performance", "MIS / SIS Support Team", 19,
     ["Portal very slow", "Student portal timing out", "System slow at registration"],
     SLOW + BROKEN,
     ["the student portal is the one affected", "it times out during the registration period",
      "many students are complaining about the portal"],
     "Identified database contention during peak registration, applied index tuning and temporarily scaled the application resources. Response times returned to normal.")

# ------------------------------------------------------------- APPLICATIONS
_ = prob("APP-INSTALL", "Application Support", "Software installation", "Applications Support Team", 41,
     ["Software installation request", "Need software installed", "Install application",
      "SPSS installation"],
     ["kindly install it for me", "i need it installed", "it is not installed on my machine"],
     ["i need spss installed for data analysis", "please install microsoft office on my laptop",
      "the statistical software is needed on the postgraduate lab machines",
      "kindly advise on the licence for the installation"],
     "Verified licence entitlement, installed the requested application from the approved software repository and activated it against the institutional licence server.")

_ = prob("APP-LICENCE", "Application Support", "Licensing", "Applications Support Team", 17,
     ["Licence expired", "Activation required", "Product not licensed", "Licence key problem"],
     BROKEN + ["it keeps asking for activation"],
     ["the antivirus says the licence has expired", "office is asking for activation",
      "the licence key provided is not being accepted"],
     "Re-registered the endpoint against the institutional licence server and reapplied a valid activation key. Confirmed the product reported an activated state.")

_ = prob("APP-ERP", "Application Support", "ERP / Finance system", "Applications Support Team", 21,
     ["Finance system problem", "ERP not opening", "Procurement system error",
      "Cannot access finance module"],
     ACCESS + BROKEN + LOGIN,
     ["this is the finance module of the erp", "the procurement system will not open on my machine",
      "it disconnects when i try to post a transaction",
      "it errors immediately after i log into the erp"],
     "Cleared the corrupted local client cache, updated the ERP client to the current build and re-established the database connection profile. Confirmed normal operation.")

_ = prob("APP-PRINT-SW", "Application Support", "Print services", "Applications Support Team", 23,
     ["Cannot print", "Printing not working", "Printer not installed", "Print problem"],
     BROKEN + ACCESS,
     ["the printer is not installed on my laptop", "i need the network printer mapped",
      "others can print but my machine cannot", "printing stopped after my system was updated"],
     "Removed the stale printer object, reinstalled the correct driver and mapped the user to the departmental print queue. Verified with a test page.")

# ----------------------------------------------------------------- HARDWARE
_ = prob("HW-PRINTER", "Hardware & Equipment", "Printer", "Hardware & Maintenance Team", 49,
     ["Printer not working", "Printer jamming", "Printer error", "Cannot print",
      "Office printer down"],
     BROKEN + ["it keeps jamming", "the output is faint"],
     ["the printer shows an error light", "the paper keeps jamming inside the printer",
      "the printouts are faint and streaky", "the printer itself will not pull paper"],
     "Cleared the paper path obstruction, replaced the worn pickup roller and ran a cleaning cycle. Confirmed clean output on a test print.")

_ = prob("HW-PC-NOBOOT", "Hardware & Equipment", "Desktop / laptop", "Hardware & Maintenance Team", 39,
     ["Computer not starting", "Desktop not booting", "PC will not power on",
      "Machine dead"],
     BROKEN + ["nothing happens at all", "it will not start"],
     ["the desktop does not power on when i press the button",
      "it beeps and stops before loading windows", "the screen stays black on startup",
      "it stopped booting after the power went off"],
     "Diagnosed a failed power supply unit following a power surge, replaced the PSU and confirmed the system booted normally into the operating system.")

_ = prob("HW-PC-SLOW", "Hardware & Equipment", "Performance", "Hardware & Maintenance Team", 29,
     ["Computer very slow", "Laptop slow", "Machine freezing", "PC hanging"],
     SLOW + BROKEN,
     ["my computer hangs when i open several programs",
      "the machine takes very long to start up",
      "i have to restart the computer several times a day",
      "the laptop itself is the problem, not the network"],
     "Performed malware removal and disk cleanup, upgraded the machine from a mechanical disk to an SSD and increased RAM. Boot and application load times improved substantially.")

_ = prob("HW-PROJECTOR", "Hardware & Equipment", "Projector", "Hardware & Maintenance Team", 25,
     ["Projector not working", "No display on projector", "Lecture hall projector problem",
      "Projector faint"],
     BROKEN + MISSING + ["there is no display"],
     ["the projector in the lecture hall shows nothing from the laptop",
      "the projected image is faint and washed out",
      "i have changed the cable but the projector still shows nothing"],
     "Replaced the faulty HDMI cable and cleaned the projector optics; where the lamp was beyond service hours the lamp module was replaced. Confirmed clear display.")

_ = prob("HW-PERIPHERAL", "Hardware & Equipment", "Peripherals", "Hardware & Maintenance Team", 21,
     ["Keyboard not working", "Mouse not responding", "Scanner not detected", "UPS beeping"],
     BROKEN + MISSING,
     ["some keys on the keyboard do not respond", "the scanner is not being detected",
      "the ups beeps continuously and the machine goes off",
      "the mouse pointer does not move at all"],
     "Replaced the faulty peripheral from stores and confirmed detection and normal operation; for the UPS the battery pack was replaced and a load test performed.")
