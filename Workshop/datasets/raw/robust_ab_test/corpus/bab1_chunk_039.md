# 3/4/24

- I’ve found that ELI5 (Explain-It-Like-I’m-5) documents are unreasonably effective at producing clarity.

  - Writing one requires you to distill a fractally complicated thing into a very simple, 2-page frame.

  - You have to distill the underlying thing to the point of caricature.

  - But the caricature is much easier for people to grab on to.

  - The caricature that everyone can grab onto provides a backbone to hang details off of that previously would have been lost in a swirl.

- Flywheels take time to get up to speed.

  - Time to bake; time for the acorn to grow.

  - If the output metric is evaluated too early or with too high a target then there are no moves to start the flywheel (or plant the acorn) that hit the target in time.

  - In that case the only move for the owner of the metric is linear strategies.

  - The solution is to either delay the timeline of evaluation or set the goals more modestly at the beginning to allow a flywheel-shaped strategy to be viable.

- Any time there’s a trade off it’s tempting to say “this side is better”.

  - But both ends are good!

  - It’s not "either, or" it’s "both, and".

  - Values statements can’t be made in a vacuum. “Speed is always better” cannot possibly be true in all contexts, because you have to contend with “at the expense of what?”

  - Even if you lean in one direction right now you want to have some weight on the other side ready to surf that way too.

    - A balanced stance with a slight asymmetry based on the current context and position of the trade off.

  - But always a balanced stance.

  - If you aren't positioned with at least a little weight on the other side, then you'll never be able to sense when you're misbalanced.

- Someone reminded me of some wisdom from the Google SRE book about satisficing vs maximizing.

  - When there are metrics everyone defaults into assuming maximization of them, but for the majority of them, you want to satisfice.

    - That is, simply clear the bar.

  - Maximizing most metrics past the good enough bar has strongly diminishing returns, and has a huge opportunity cost.

  - Set a “green zone” target on a metric.

    - You may have meetings to set this target.

  - Then, once you reach the green zone, you stop working on it.

    - This should happen automatically, no need for a meeting.

  - You can only set the bar when you’re sober.

    - When you're responding to an incident and adrenaline is coursing through your veins you aren't sober.

- Perfect plans are impossible in uncertainty.

  - And uncertainty is *everywhere*.

  - “If only we had taken the time to do the perfect plan, we could have avoided this expensive clean-up work!”

  - But the perfect plan is impossible.

  - The perfect *meta-*plan is resilient to many possible real world outcomes. It *assumes* uncertainty.

- A very large, concrete idea will often feel abstract.

  - Our signal for if an idea is abstract is how easy it is to “grab onto” via concrete details or concrete touch points.

  - A very large idea that is composed of a swarm of small concrete ideas will *feel* abstract; there’s no good place to “grab onto” the whole idea.

- Everyone thinks they're special.

  - And you are special… to you!

  - But you're not globally special unless you can convince others who don’t have a vested interest in you that you're special.

- It's impossible to design a perfect API in a vacuum.

  - A useful API is a negotiation, a mutual process of discovery with your ecosystem.

  - This property means that it is fundamentally uncertain.

  - If you wait until an API is “perfect” to ship, you will ship nothing, or you will ship a thing that if you get even a single detail wrong it will not work.

  - You have to ship an API that is good and useful and then clean it up and evolve it in response with the ecosystem.

- Prioritization cannot be done in a vacuum, it must be done in context.

  - You can't make a prioritization decision without considering trade offs.

  - "Do we care about Germany?" is not a coherent question to ask.

  - "Do we care enough about Germany to not do X, Y, Z" is the real question.

- “Every system is perfectly designed to get the result that it does.” - Deming

  - Some systems cannot adapt, they are machines.

  - A machine is perfectly designed for the thing it actually does.

  - A living thing has the agency to change itself.

- Whenever your prediction turns out to be wrong, seek to understand why the prediction was wrong.

  - The amount of effort you invest in understanding it should scale with the size of the surprise and how likely you think you’ll be in a similar kind of situation in the future.

  - Was there some signal available at the time of prediction that you paid too little (or too much) attention to?

  - How is it similar or dissimilar from predictions you’ve made in the past?

- It’s easier for organizations to coordinate around obvious answers.

  - That means that when there isn’t a single silver bullet answer, it’s harder to coordinate.

  - For example, if it’s a game of inches with sub-linear returns on each effort until you reach critical mass.

- A general design characteristic for things that have potential privacy or security implications: minimize the chance of a nasty surprise.

  - Show a proactive indication if the system did a thing that has privacy implications, so if the user thinks "wait why is that showing up right now, I don't want that" they can proactively discover it now, not passively discover it later and have an "oh crap" moment.

  - OSes do this now with system-level microphone / camera indicator.

  - ChatGPT’s new personal memory feature *doesn’t* do this.

  - It’s possible to ask it to save little memories about you (e.g. “I typically program in Typescript”).

  - But it turns out the system also saves things it *thinks* are important bits of context for you.

  - This can lead to nasty surprises. A friend found “I like cacti” in his memories, but it could have been something significantly more embarrassing.

  - A way to improve this UI: every time the system stores a memory from a message, show a little icon next to the message allowing the user to inspect the memory and delete it.

- Under the right conditions, if you can accept 20% more chaos, you can get 200% more output.

- It’s not so much that the prophets of disruption within an incumbent are ignored.

  - It’s more that as the incumbent in the face of disruption, even once you recognize the threat, it’s not obvious what to do about it.

  - There are often no good moves–any given disruptive play is unlikely to work, especially because it’s a threat to your main businesses and will go against the grain.

  - Since there are no good moves, the org does nothing.

  - The org is frozen in place, staring down the steamroller, hoping it will stop before it reaches them.

  - This is why disruption is an asymmetric threat.

- Agility and velocity are in tension.

  - In a bigger organization it's not just about you, it's about the ripple effects of your actions in the vast fabric you're a part of.

  - Your desire to have more agility and autonomy is in a way selfish to the other thousands of people in your org who have to be able to figure out what you're doing and do something that coheres with it.

  - Do you want local fast or global fast? Global fast will be locally slower.

  - My friend Dimitri recently distilled this nicely in [<u>Aircraft carriers and zodiac boats</u>](https://whatdimitrilearned.substack.com/p/2024-02-26).

- Last week I learned about [<u>Boundary Objects</u>](https://en.wikipedia.org/wiki/Boundary_object).

  - "In sociology and science and technology studies, a boundary object is information, such as specimens, field notes, and maps, used in different ways by different communities for collaborative work through scales. Boundary objects are plastic, interpreted differently across communities but with enough immutable content (i.e., common identity across social words and contexts) to maintain integrity."

  - Boundary Objects are the way distinct groups of people that speak different languages create shared meaning together.

- For something to be alive it has to escape the control of its creator.

  - Memes that escape their original creator are viral; alive.

  - A scenius is a collection of people who escape their catalyst.

  - The lack of control of the creator is precisely where the super-linear value emerges.

  - The mess is the point of a living system; it is the upside.

- A friend recently asked me for two exhortations that I wish everyone in the world could wake up with firmly implanted in their minds. My answers:

  - 1\) Compassion is strength

  - 2\) Your need for certainty is holding you back.

- Some frames preclude there even being an answer.

  - The question is often more important than the answer.

  - Detecting if it’s a good question:

    - Is it generative?

    - Does it give you a scaffolding for creating a shared understanding?

    - Does it help you make better bets?

    - Do many different people find that it clarifies a key dimension?

- If you think something is crystal clear but it’s actually murky, you’ll make reckless decisions.

- If you assume the magic happens discontinuously you'll miss it when it happens.

  - You'll be looking for an obvious, impossible-to-miss miracle.

  - Real magic tends to happen continuously, slowly.

  - In the moment, real magic looks like luck.

- The entity that tells the best story gets the most power.

  - Power is largely based on the beliefs of others about who has power.

  - People then make decisions on how to act based on who they believe has power, which gives the people they believe have power actual power.

  - Viral, interesting, distinctive stories spread further and transmit more power.

  - In *Nonzero* by Robert Wright, he talks about the game theory of over-the-top punishments in a world of chiefdoms.

    - When a given chief vanquishes a rival, if they do it in a memorable and over-the-top way, the story is more likely to travel.

    - “Did you hear about the chief who \[did some terrible, over-the-top thing to his enemies\]? Can you believe it? If it’s true, I’d never want to cross him…”

  - The story is naturally viral. People feel compelled to share it because it’s interesting and shocking.

  - It’s not even necessarily that chiefs did this on purpose; it’s just that the ones who did had their stories spread wider, and therefore were more powerful, and that phenomena became more common.

    - The ones who didn’t do anything spectacularly over the top faded into obscurity.

  - [<u>Pirate ships worked the same way.</u>](https://www.youtube.com/watch?v=3YFeE1eDlD0&ab_channel=CGPGrey)

  - These examples are horrifying and immoral, but the phenomena is amoral. A similar virality occurs in other contexts as long as it’s a good story.

  - “Did you hear about the founder trying to raise \$7T? Can you believe it? Maybe he knows something I don’t…”

- People naturally invest time and effort in the entities and ideas they suspect will be powerful.

  - For example, currying favor with people they think are powerful or might become powerful.

  - It’s a kind of emergent “yes, and” survival of the fittest / most powerful.

  - It’s not so much that this powerful person creates the wave they surf.

  - It’s that the surrounding population sees their initial ripple, decides to bet on it, and rushes to it.

  - That extra energy creates a larger wave, which attracts more energy.

  - This can be a compounding loop for a thing many people believe is powerful.

  - But compounding loops run fast in both directions; such a leader suffering a humiliating defeat will lose many followers, which will start a spiral in the other direction.

  - Such a leader will avoid humiliation as though it was death.

  - This is one of the reasons for an ecosystem or emergent human system the perception of momentum is so important.

- Weak leaders think the main thing to optimize for is to do things that are perceived as bold.

- Weak leaders want all of their subordinates to be just like them.

  - To have the same strengths… but also the same blindspots.

  - Strong leaders love it when their subordinates have strengths that complement their own.

  - The most valuable strengths are the ones that most complement the strength gaps of the team.

- When you’re using an asymmetric playbook, the traditional players won’t understand it at all.

  - “What are you doing? That looks extremely wrong.”

  - They literally won’t see it until it encircles them.

  - Of course, it’s difficult to distinguish an effective asymmetric playbook and a bad one until after you see if it works.

  - But once it does work, it will look obvious and inevitable in retrospect.

- When you don't fear death in a given context, you can play your own most bold and audacious self.

  - If you know there's no game-over condition, you can swing entirely for home runs.

- A magic trick of foresight in organizations using techniques of cold reading.

  - Form a hypothesis about a thing that might happen

    - "this team will not like working with this other team because the other team is good at prototyping and they're more of a platform team"

    - "this area of the org will be reorged soon."

  - Base your hypothesis on a savvy summarization of the priors in the space.

  - Then try to find little snippets of disconfirming evidence.

  - Look for it everywhere.

    - In every 1:1 poke and prod (lightly and delicately) for it.

    - Even amorphous statements like "Sure is a lot of change happening!" will often solicit disconfirming or confirming evidence.

  - If you don't find any disconfirming evidence at all then that's a good sign that your hypothesis is right.

  - The less disconfirming evidence you find, the more you can cryptically plant hints. "I don't think we should assume that org structure is stable."

  - These are hints that if the hypothesis is correct will make you look like a wizard, but if they don't hit, no one will notice.

  - If people think you're smart and plugged in and have a good track record, they'll fill in any ambiguity in ways that are in your favor.

    - So even if you only know 1/3 of the reorg shape, if you keep details vague, people won't realize you only had a general vibe and will think you knew all of it.

  - This then makes people more likely to trust your vague predictions in the future, and also more likely to gossip with you to trade subtextual information, which helps you make significantly better predictions.

  - The asymmetry of this works because you need only one sliver of disconfirming evidence to disprove the hypothesis, so if you're searching for them and don't find it, it's likely to be true.

  - People don't want to lie, so if they know the information they might say things that individually aren't revealing, but they won't disconfirm the implied hypothesis if it's right.

  - If there's a secret that a lot of people know, there is information evaporating off of every interaction they have.

  - It's extremely subtle, but you can sense it if you suspect it might be there and blur your eyes just right, like a magic eye painting.

- Dysfunctional orgs put leaders in impossible situations where the only thing they can do to survive is bad faith tactics, e.g. spiking any projects that would possibly undermine their kayfabe to anyone.

  - In such a situation if you act with integrity you are knocked out of the game.

  - So you do some bad faith tactics to stay alive: "I'm just doing this for now, to get to a stable situation where I can act with integrity".

  - But it never ends, you have to do more and more.

  - And meanwhile people watch you work without integrity and think about you "I'd be happy if they were no longer here, they're clearly not acting with integrity."

  - Deeply dysfunctional orgs irreparably taint the leaders who are in them and make them hollowed out husks.

- No one will use another person’s hyper-bespoke situated software.

  - For example, everyone loves their own complex spreadsheet but hates everyone else’s.

  - But people will reuse other’s building blocks.

  - If you’re searching for just the killer app for the ecosystem, you’ll miss the killer building blocks.

- The holy grail of assistive apps has long been a collaborative trip planning app.

  - But no one has ever successfully constructed such an app. Why?

  - Because there are too many bespoke combinations it would have to cover

    - Every airline / travel partner integrated.

    - Every chat app / ad hoc friend group communication style.

    - Integrating various notes and goals, from semi-structured requirements to notes on scraps of paper.

    - Every friend has to agree to use the same app.

  - That is, everyone has a slightly different workflow they use to collaboratively trip plan today.

  - If you make an app that perfectly solves a given user’s workflow, the TAM is a single user: Far too small to justify building it.

  - To solve this problem, you’d need to atomize apps into smaller components that could be assembled into a just-in-time bespoke workflow.

  - A swarm of self-assembling situated software.

- A lot of things seem to run in ~30 year cycles.

  - Especially for fashion and nostalgia.

  - But also perhaps for technical paradigms.

  - Perhaps it has something to do with the length of a typical person’s productive adult working career?

- Apparently the Roman notion of property boils down to: are you allowed to destroy it?

  - If so, then it's yours.

  - This is also the logic behind “it’s easier to trust technology you can throw out the window”

- We live in the era of aggregators.

  - There’s only aggregators, and temporarily embarrassed aggregators.

  - No one can imagine anything else as being worth shooting for.

  - Even the entities that are most focused on “arming the rebels” look for opportunities to become an aggregator themselves.

  - Personal AI is either going to be the apex aggregator, or the end of aggregators.

- Aggregators seem impossible to beat because of their power of distribution.

  - But internally they are impossible-to-coordinate, lumbering beasts.

    - Ask anyone who has ever worked in one!

  - If you could do an asymmetric play that disrupts their advantage, you might be able to scramble their logic and outmaneuver them.

- We want to own our narrative of who we are.

  - Famous people lost this ability a long time ago.

  - But as the internet continues to speed up it's coming for the rest of us too.

  - The time honored tradition for people who have become famous for a bad reason (e.g. an embarrassing story) is to generate lots of neutral to positive stories to drown it out.

  - This is like sending out a cloud of chaff to obfuscate the story you don’t want others to hear.

  - AI now makes generating this kind of chaff significantly easier than ever before.

  - This will create a background noise level that is cacophonous, and finding real information will get increasingly difficult.

  - In some ways, this is postmodernism to the terrifying extreme: not just “we can’t sort through which things we agree are true” but “actively generating false content to signal jam the real world.”

- People don't buy a brand because it's good, it's because they know the floor.

  - By buying from a brand you've capped the downside.

- Having taste is a liability in large organizations but a huge advantage everywhere else.

  - Taste is the ability to cut to the core of a complex topic with an opinionated and caricatured but resonant and self-evidently true observation.

  - Taste and kayfabe are like matter and antimatter.

  - Taste is necessary out in the world on your own.

  - It's only in a protective cage of a successful organization that kayfabe is implicitly encouraged and considered useful.

- Taste is the final moat.

  - In a cacophonous world of abundance, good taste is the scarce commodity.

  - Taste is only valuable to the extent it stands out from the baseline.

- If I had access to a time machine, what would I do?

  - First I’d go back in time and stop Hitler, obviously.

  - But high on the list would be going back to 2007.

  - I’d storm the stage when Steve Jobs was introducing the iPhone.

  - “I’m from the future and this device will become the single most important computing device in the world. For the love of god do not allow Apple to only run apps they sign, that’s insane!”

- I’ve asserted in the past that knowhow is fundamentally impossible to transmit between humans.

  - I think it’s partially related to the second law of thermodynamics.

  - As a signal propagates through a transmission medium it diffuses.

  - This means that within a transmission medium, signal propagation is a broadcast.

  - This quickly creates a cacophony, so you must create a boundary to prevent the signal from propagating too widely.

    - For example, you create an insulated wire and transmit the signal through it; the signal is a broadcast *within* the wire but does not escape the wire except at the ends.

  - Boundaries, to be effective, must only allow a small fraction of the internal signals to exit.

  - If they didn’t, it would be a cacophony and impossible for any signal to be heard above the background noise.

  - The more narrow the pipe to another system, the more signal that has to be elided, perhaps many many orders of magnitude.

  - Knowhow is a complex set of states maintained inside of our brains.

  - Knowhow can only be transmitted at the rate that someone can speak, at the limit.

  - That’s a teensy tiny straw to pass ideas through.

- AI can fake a lot, but it can’t fake a cryptographic signature.

  - So you can use cryptographic signatures as a building block for a credible and safe internet in the AI era.

- Everyone's acting like AI is a sustaining innovation.

  - Where's the fun in that?

  - Let's assume AI is a disruptive innovation, and act like it!

- Software used to be exciting.

  - Nowadays there’s a New York Times article when Instagram changes a button.

  - It’s slow! And boring!

  - The speed of innovation is tied to where in the s-curve you are.

  - Getting excited again will require the irruption of a new paradigm.

- A desirable property of open systems: not decentralized, but decentraliz-*able.*

  - Decentralized systems are extraordinarily hard to change, so if they start big you had darn well better get all of the interfaces between components right!

  - Decentralization is expensive, so don't pay for it at the start if you don't need it.

  - Instead just design your systems and incentives so as it gets momentum, it will naturally get more decentralized.

  - A decentralized-able thing can grow and the interfaces can evolve as they grow and become more decentralized.

  - Ensure your system is minimally-viable-decentralized to start, and has clear buds of decentralization to bloom as the system does.

- One of the best positions to be in in an open ecosystem is an optional but central complement to the ecosystem.

  - This is especially true if you have network effects that cause your quality to go up with more users.

  - Users are willing to use you even when you’ve grown big, because you don’t have undue formal leverage.

  - You must actively compete on quality to maintain your perch.

  - But because of the network effects, it gets harder and harder for others to compete.

  - The result is you’ve earned the right to be the king of an important hill.

- One of the superpowers of the web: speculative execution.

  - When you click a link, before you see any permissions prompt, the content can be fetched and executed… safely.

  - This is the ability that allows the zero-friction “teleportation” between sites.

  - Reducing friction of first use by orders of magnitude enables a level of virality impossible for all but aggregators in other paradigms.

- The cold start problem for network effects shows up because of friction of first use.

  - E.g. “Do I want to trust this random startup with all my data?"

  - In the current paradigm, the marginal user has to decide if they trust a startup they've never heard of before with their life.

  - Something that tweaked this friction and allowed safe speculative execution could change a force of gravity.

  - This would radically change the intuition for how things work.

- When there’s a lot of capital flowing into a new area in a frenzy, competition is intense.

  - Everyone is fighting hard to find toeholds that might then balloon into massive businesses.

  - To compete, you’ve got to be a bigger shark than the other sharks you’re in the tank with.

  - This is exhausting and hard!

  - Another approach: sit on a raft above the frenzy.

  - Build a thing on top that *assumes* that some number of sharks in the tank will be successful.

  - You don’t need to know which ones will be successful for the technique to work.

  - You’ll also have less competition because you’re a step ahead of everyone else.