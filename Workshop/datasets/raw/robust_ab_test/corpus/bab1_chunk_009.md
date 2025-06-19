# 10/7/24

- A tactic people use to avoid algorithmic censorship: “Voldemorting”

  - You talk in ways that will be very clear to the listener about what you mean, but that are oblique enough that the algorithm doesn’t notice.

  - Apparently in China people will find characters that *look* like the word you’re trying to say, but don’t *sound* like it.

  - But communication only works if the receiver can recover the intended signal, so these indirections have to be obvious enough for a receiver to understand.

  - And particularly effective he-who-must-not-be-named phrases will become used more often (since they effectively communicate the intended message)... thus making them easier for an algorithm to detect and search for.

  - That means that the replacement phrases must constantly be evolving.

- The more efficient the communication between two entities, the more unintelligible for onlookers.

  - Efficient communication requires a shared context; information that can be presumed for both the sender and receiver and thus factored out of the signal.

  - That allows significant compression… but at the cost of making it unintelligible if you don’t have the context.

- LLMs “talk” to themselves internally in embeddings, and only reduce to words when they need to talk to humans.

  - (This is a highly stylized metaphor of the actual workings of LLMs.)

  - When an LLM needs to communicate with another entity, it reduces to words, which can be understood by a human or another LLM.

  - But imagine if two systems are speaking, and have the same LLM on both sides.

  - If it can discover that, it can talk in the more efficient encoding: in the embedding, directly.

    - This requires both the sender and receiver to be using the same embedding space.

    - A natural network effect for the embedding space; people will tend to use the one that others use, all else equal, to have a better chance of communicating with a partner.

  - But now an observer will feel left in the dark.

    - “What are the two of you talking about? … You aren’t plotting anything, are you? … Hello!?”

- Training gives LLMs background knowledge. Context gives them working knowledge.

  - Many people are worried about LLMs using their data in training, but there the leverage is way lower for the model creator.

  - Your data is just an anonymous particle of sand in a sandstorm.

  - It turns out that the value of the “querystream” as a data source is just not that interesting compared to all of the other data streams to feed LLMs, as evidenced by how willing the major model providers are to agree to contracts to not use any queries for training.

- The podcasts from Google’s NotebookLM are scary good.

  - I know I’m late to the party on this, but I finally fed in some of my writing and got [<u>this podcast</u>](https://notebooklm.google.com/notebook/29d440f5-5b24-49be-b947-5e6c04d3520c/audio).

  - I think this is arguably the most clear podcast related to my writing and ideas ever–it even discovered new metaphors and ways to describe the key ideas.

  - Over the weekend I spent a few hours throwing rich content at NotebookLM that I wanted it to digest, and found new insights every time.

  - A new indispensable tool in my toolbox!

- AI will unbundle the app.

  - Why are apps the size they are today?

    - Why do we have the “chunkiness” of the app, in terms of the number of use cases that are typically bundled into one app?

  - Part of it is because software is expensive to write, cheap to run.

    - High fixed cost, low marginal cost.

  - Over time prices tend to reduce to the marginal cost of production.

  - Over sufficient time, with sufficient competition, software thus reduces down to ~free, but is only created if there’s a large enough market to justify the fixed cost of creating it.

    - Caveat: high switch costs effectively reduce competition and slow this inevitable margin compression.

  - But LLMs make software much cheaper to create.

    - For small, non-generalized software, the cost approaches zero.

  - When the fixed cost of writing software goes to 0, the minimum viable market for a bit of software also approaches 0.

  - That implies that we’ll see an explosion of software that is so crappy and bespoke to a particular use case that before no one would have bothered to build it.

    - Marvelously powerful and situated and niche.

  - When software has a high fixed cost, the use cases have to congeal into a bundle that is big enough to have a market to make the bundle viable… and that also contains within it a business strong enough to at least cover the fixed cost.

    - But when it’s zero fixed cost, those bundles no longer need to coalesce, and they also don’t need to have a business model inside: the fixed cost is zero, there’s nothing to pay for!

  - Software today does not actually have zero marginal cost: it has a distribution cost.

    - The equilibrium point of “chunkiness of app” that we see today is based on the fixed cost of software production and the marginal cost of distribution.

    - AI reduces the fixed cost of production.

    - An alternative set of laws of physics could reduce the marginal cost of distribution.

    - Together it could lead to wildly different outcomes.

- With traditional software, you have to hope some PM somewhere prioritized the feature you want.

  - That the PM noticed that such a feature would be useful, and was worth spending considerable effort to build.

    - That effort goes up considerably as the team size gets larger: far more people to coordinate and to get to agree the idea is important enough to do versus all of the other things they could be doing.

  - The only time this happens is when *lots* of other users have the same basic need as you.

  - This phenomena happens today because software is expensive to write, it has a high fixed cost.

  - But as the fixed cost of software goes down, it will get to the point where you can write your own software, or have an LLM create it for you.

  - You won't have to cross your fingers that some PM somewhere proactively thought of a thing that you find valuable!

- [<u>OpenAI apparently has 10m paying subscribers.</u>](https://stratechery.com/2024/openai-devday-openais-wrenching-transition-lonely-at-the-top/)

  - An impressive number!

    - Convincing consumers to pay for something, especially a new kind of thing, is extraordinarily hard.

  - But in the future most consumers will have to have an LLM subscription of some kind.

    - The applications of LLMs will continue to grow, becoming a thing you couldn’t imagine living without.

    - LLM inference is too expensive to be supported by advertising.

    - In the future most consumers will have an LLM subscription.

  - 10M users out of 7B doesn’t sound as big.

  - At the beginning of the internet, one way to get online was America Online.

    - AOL gave access to the open internet, but also to the proprietary content and chat rooms.

    - At the beginning, the open internet wasn’t that useful, and the value of the bonus proprietary content was the most important.

    - But as the internet’s natural compounding loops took off, the relative value of the proprietary AOL content declined and was ultimately surpassed.

    - Imagine a consumer subscription as a thing that gets you an LLM subscription… but you can only use it in the proprietary closed world of that provider.

    - An offering that allows you to use your subscription in an *open* ecosystem of applications would be far more powerful.

- Complex systems are impossible to understand with reductionist tools.

  - But that’s not to say they are impossible to understand.

  - They can be understood with a different set of tools.

  - The best way is to gain *knowhow*.

  - Direct experiential knowledge, informed by hands-on interactions with the system, that helps congeal a gut feel.

  - When you have knowhow, you can’t describe *why* a given decision is right… but nevertheless you have a success rate significantly beyond what chance would predict.

  - Saying something is chaotic absolves you of the ability or responsibility for developing knowhow, an edge.

  - Instead of giving up, dive in.

  - As you engage with the system, you earn experiential knowledge, and knowhow.

  - Knowhow is a real edge in understanding a complex system.

  - Fingerspitzengefühl: the feeling, the knowhow that comes from practice using the thing.

- The process of becoming an expert in something is about absorbing lots of experience into your mind’s System 1.

  - System 1: cheap and fast but very specific matching.

  - System 2: expensive and slow but general.

  - When you have the relevant experience in your System 1 associative memory, your System 2 doesn't have to be engaged as often.

  - Your System 2 is great at using reductionist tools; that means it’s terrible at understanding complex phenomena.

  - You have to "burn in" all of the possible situations as experiential memories, to be available for the System 1 to later draw on.

  - If there's something unlike what you've done before, you have to engage your System 2, and it will be hard and error prone.

  - Practice is like taking in all of the angles and remembering them so you'll be able to recognize them automatically later.

- LLMs need to be understood via your System 1, not your System 2.

  - LLMs are inherently complex and hard to reason about.

  - The only way to be able to use them effectively is to develop significant experiential knowledge with them: knowhow.

  - Engineers are most familiar with using the CS lens to understand a problem.

    - But CS is fundamentally a hyper-reductionist lens.

    - It cannot be used to understand complex phenomena.

  - I can’t tell you how many extremely smart engineers I know who have endeavored to “understand” LLMs by building one themselves.

    - This takes *months* of careful study and experimentation, and at the end you get a crappy little model that is orders of magnitude worse than the leading models.

  - The key question with LLMs is not how they work, but *what* you can use them for.

    - The latter question is impossible to answer with CS, especially for a non-ML expert.

  - The only way to answer that question is to actually play with them, deeply, and often.

  - How an LLM works gives you zero insight into how to use them.

  - To become an LLM wizard, you have to develop the knowhow.

- Someone with more knowhow in a context will be like a wizard.

  - How does it work?

  - They have loaded up their System 1 with more relevant experience.

  - But also they’re able to “chunk” memories at a higher level.

  - Grandmasters of chess can look at any chess game and remember it in a second or two.

    - But if you show them a *random* board (i.e. not one generated by a legal game) they are no better than a novice at recalling the positions of pieces.

    - The grandmaster is chunking the knowledge of the board at a much higher layer of abstraction.

  - People who are able to work at higher layers of abstraction will be more effective with LLMs.

- People who know how to use LLMs are wizards.

  - They can use LLMs extremely effectively, to do things that look like magic.

  - Using an LLM effectively is a form of knowhow.

  - Knowhow doesn’t *feel* like expertise, so when someone asks you how you do it, you’ll shrug and say, “simply do what I do, it’s easy!”

    - But it’s *not* easy!

    - It feels easy to you as the person with knowhow because you’ve spent hundreds of hours practicing, loading up the experiences into your System 1 so they can be effortlessly recalled with minimal effort.

  - As people get more knowhow to use with LLMs, the difference between the average user and the best user gets stronger.

  - How can we make it so more people can be LLM wizards?

  - How can we make this magic common?

- Prompting is like casting spells.

  - You have to get them right to get them to do what you think they will

  - The more knowhow and practice you have, the more likely it is to work.

- Guillermo Rauch [<u>points out</u>](https://x.com/rauchg/status/1840293374059839726?s=51&t=vzxMKR4cS0gSwwdp_gsNCA) that with LLMs sometimes it’s faster to create software than to Google it.

  - What a time to be alive!

- The easiest way to “finetune” a model is to curate the context.

  - LLMs with a well-curated context will be *significantly* better at producing outputs that are useful.

  - The tools that create the most value with LLMs will have UX innovations to help even less-savvy users drive LLMs to useful outputs easily.

- LLMs are not the thing; they create the conditions for the thing.

  - They disrupt the cost structures, enabling the creation of a new thing.

  - Back before the printing press, the cost of book production was massive.

  - The people who illuminated manuscripts knew how expensive it was to produce books, and that’s why only a small number of extremely important books could be produced, and those books could rarely change.

  - But then the printing press came about.

  - Suddenly things like daily local newspapers were now possible.

  - The illuminator would look at the amateur newspaper publisher and have no idea how it happened.

  - “Don’t you realize that’s impossible? How are you doing that?”

  - “... idk, I just turn the crank and copies of the newspaper come out?”

- Remember when you were a kid and the web was magical?

  - The web felt more participatory, not a passive thing to consume.

  - It was wonderful and wide and weird.

  - Let’s bring that magic back!

- All new things come from deviating from the norm.

  - A defection that turns out to work is acclaimed.

  - But the vast majority of the time, they don’t work.

  - Sometimes the deviation is accidental.

  - But sometimes it’s intentional: a *defection*.

  - The renegades are where innovative ideas come from–but also where a bunch of junk comes from.

  - High beta.

- New creations are speciation events.

  - They need to deviate from the core, from the status quo.

  - The more signals that are being spread around, the more quickly it pulls everyone to the status quo.

  - You need a bit of isolation for speciation to occur.

- [<u>Al Pastor was discovered by Lebanese immigrants in Mexico</u>](https://adhc.lib.ua.edu/globalfoodways/carne-al-pastor-a-mexican-national-dish-straight-from-lebanon/).

  - But it was waiting to be discovered; it's implied by human taste.

  - What other vibes exist that are not yet discovered?

- Consensus cannot do innovation.

  - Consensus pulls to the average.

  - Consensus doesn’t allow the isolation that speciation needs.

  - Innovation is a defection from the average that turns out to be valuable.

  - Therefore consensus can’t create something innovative.

  - If you want to create something innovative and novel that will ultimately be a new standard, it can’t come from a pre-existing standards body.

    - The body will take any new ideas and pull them strongly to the consensus before they have a chance to take hold in the real world.

  - Instead, you have to create it, and then as it gains momentum and others choose to join with it and participate, it gains even more momentum.

    - Of course, the vast majority of these experiments will fail!

  - The standardization momentum thus accumulates *after* the original innovative act.

- A useful concept: MAYA

  - Maximally Advanced, Yet Acceptable

  - There’s only so far beyond what is currently acceptable that people are willing to go.

  - The further from the status quo, the more you stick your neck out to do it.

    - The further from the status quo, the less likely it is to work.

    - The further from the status quo, the more costly it is.

      - Both to execute it, but also the recovery cost if it fails, which is inherently higher.

    - You might also have to explain your actions to everyone doing it the normal way.

      - That can be annoying, but in some cases it can also lead to actively being punished.

  - MAYA is the threshold that sets the bounds of the iterative adjacent possible in what kinds of ideas might work.

  - Things that are beyond the MAYA threshold are noise; impossible to cohere.

- What counts as MAYA has to do with the background context.

  - For example, if you tried to introduce jazz in 1850 it would have been rejected.

  - Jazz is a speciation event in a coevolutionary space.

  - It required a certain background context to emerge from.

  - An [<u>essay from TS Eliot</u>](https://socrates.acadiau.ca/courses/engl/rcunningham/Winter2020/engl5013_poetics/texts/eliot_tradition.pdf) on artistic genius depending on a background of tradition, of shared context.

- Apparently a famous musician had an interesting tactic for developing new ideas.

  - He’d pay a bunch of different amazing musicians to come to the studio.

  - He’d put them in individual rooms and tell them to just explore and create.

  - He’d walk between the rooms, listening to find ideas he liked, perhaps giving small nudges to them about things to lean into.

  - Then later he’d curate his favorite bits into his own novel synthesis.

  - He got a super-linear quality increase.

  - Instead of being limited by his own time and effort, he could parallelize it, and then use his *taste* to select from a larger number of streams.

  - It was expensive to do, and was only possible because he was so rich.

  - But now anyone savvy enough can do it with LLMs!

- LLM output is kind of like the tyranny of marginal information.

  - When you take in the entire corpus of humanity you get the middle of the distribution.

  - Will LLMs help people find new ideas / vibes?

  - Or do LLMs just drive to the average of everything?

  - The LLM is un-opinionated and bland.

  - But where you *drive* it is not.

  - LLMs are great concept colliders and help make sense of the random things you did.

  - LLMs can generate, but humans select what's useful out of it.

  - The human judgment of what to drive the LLM to do is what gives it meaning and value.

- LLMs can be used to create, but also to curate.

  - Because LLMs make so much slop, you need to filter.

  - An arms race where the creator and curator side need to both use AI to keep up with it.

- Consumption and creation are two fundamental forces.

  - If you are consuming more than you’re creating, you’re a passive participant.

  - How far off the ratio is is a measure of how passive you are.

  - Creation is far, far, less common than consumption in almost all things.

    - Creation is orders of magnitude more expensive than consumption.

  - Sometimes we create in one context, but are passive consumers in others.

- Clay Shirky wrote a book in the early 2000’s called [*<u>Cognitive Surplus: Cognitive Surplus: How Technology Makes Consumers into Collaborators</u>*](https://www.amazon.com/Cognitive-Surplus-Technology-Consumers-Collaborators/dp/0143119583).

  - The thesis was about a new wave of creation.

  - TV was an inherently passive experience.

    - You tuned to one of a limited number of channels, and then plopped down and images and sound were beamed into your eyes and ears, with no interactivity.

    - Hyper passivity.

  - The late 90’s was a depressing time, dominated by passive consumption and TV.

  - But the internet opened up a new venue for interactivity, for creation.

  - You could create a webpage… and even if you didn’t do that, you *decided* where to visit, what to engage with, out of nearly infinite options.

  - What would humanity achieve with this new cognitive surplus?

  - (Note that as TV stopped consuming *all* of our hours, TV didn’t go away… in fact, TV’s golden era of prestige TV occurred after the Internet had gotten big)

  - We’re in a similarly depressing era today as the late 90’s.

  - Everyone’s attention is absorbed by the infinite feed.

  - The infinite algorithmic feed is like TV in its passivity, but even worse.

  - The infinite feed is hyper-TV.

  - Instead of one-size-fits-all content of TV, it’s infinitely customized to hold your engagement in particular, in a maximally passive stance.

  - Needing to create is costly and hard, so for a service with a feed to maximize engagement they want to make it as easy as possible: just lean back, and let the algorithm figure out how to keep you engaged.

  - Amusing ourselves to death.

  - The internet was a disruptive technology that invited creation, a return to a more active stance.

  - LLMs are a disruptive technology, too.

  - Let’s use them to help encourage creation and human agency again.

- There’s a difference between an organization selling Coachella tickets vs Burning Man tickets.

  - In Coachella, the entity producing the event is producing the vast majority of the things that you’re paying to see.

    - The event could be great if they do a great job, or bad if they do a bad job.

    - Even if the other participants are passively consuming and not adding much, it could still be great.

  - In Burning Man, the entity producing it is providing the infrastructure for great things to emerge.

    - E.g. securing land permits, securing baseline infrastructure, coordinating dates for the event, etc.

    - But all of the greatness at the event comes from what the participants create, emergently.

  - The former creates an experience that could be largely passive; the latter creates an experience that must be largely active.

- I’m reading Harari’s new *Nexus* book and I love it.

  - I’m only a third of the way through currently.

  - One big idea is the tension between order and truth.

    - They can co-occur, but they are actually orthogonal, and often at odds.

  - In one of the chapters I just read, he talks about the auto-catalyzing behaviors unleashed by the *Malleus Maleficarum.*

    - It was a particularly virulent idea.

    - The techniques it proposed for interrogating a suspected witch guaranteed that you’d get a confession.

      - If the suspect didn’t confess, you tortured them in heinous ways until they did, and when they confessed, you killed them.

    - This guaranteed that every suspicion would find confirming evidence.

    - Often in the confession they’d be tortured until they named accomplices… which were a fresh round of suspects to go torture and find even more confirming evidence.

    - As the virus spread in people’s minds, the scale of the confessions being uncovered showed that the problem was a significant and large-scale one, meaning it was incredibly important.

    - Anyone who tried to stop the madness or question the tactics would be labeled a sympathizer and tortured.

    - A collective delusion that was auto-catalyzing and led to extraordinary numbers of horrendous deaths.

- When you apply high modernist lenses to a complex system it won’t work.

  - If you say it's chaos then there’s nothing you can do.

    - Diagnosing something as chaotic absolves you from the need to predict what indirect effects your actions might cause, because it’s unknowable.

  - Computer science is a hyper-reductionist view.

    - *Extraordinarily* powerful for complicated phenomena.

    - But has nothing to say about complex phenomena.

      - Well, other than computer science being useful to run simulations to get a handle on complex phenomena.

  - Something an engineer once told me: “Things that can’t be understood by computer science either are fundamentally unknowable or don’t matter.”

  - And yet complex problems are arguably the most important for us to grapple with as society.

  - No other lens is as powerful as CS is for complicated problems, and yet a diversity of less-powerful lenses together can help us develop perspectives that are significantly more likely than random to be useful.

- I love Anthea Roberts’ concept of [<u>dragonfly thinking</u>](https://www.anthearoberts.com/dragonfly-thinking).

  - Any mental model we apply to a problem is a lens.

  - A lens must reduce the signal of the real world into an easier-to-consume distillation.

  - Lenses are great, because they help us extract new insights from the fractal complexity of the real world.

  - The danger is if you use a single lens exclusively.

    - Imagine fusing rose colored glasses to your eyes permanently.

    - You’d have a dangerously skewed understanding of the world.

  - That’s why it’s critical to use a *diversity* of lenses to understand complex phenomena.

  - Dragonfly eyes have myriad lenses in every direction.

  - In Anthea’s metaphor, that gives you a multi-faceted understanding of the world around you.

  - This meta-approach is the best one in complex domains.

  - Doing this kind of translation takes mental effort; when we’re mad or scared or stressed we don’t do it.

  - This is something that AI should in theory be able to help with the heavy lifting of, giving all of us more robust, nuanced understandings.

  - Looks like that’s precisely what Anthea’s working on: [<u>https://www.dragonflythinking.net/</u>](https://www.dragonflythinking.net/)

- You know you’re in a complex space if everyone is mad but no one agrees on what to do to fix it

  - There are no improvements, there are only trade offs.

  - Every new solution creates a new wicked problem.

    - But at least it’s a new one!

    - A spiral, not a loop.

    - Progress in one dimension, churn in another.

- Every system has a self-righting zone and a self-intensifying zone.

  - Some systems are convex: self-righting.

    - When the system is perturbed out of equilibrium, the internal forces naturally pull it back to equilibrium, with the magnitude of force scaling with how far out of equilibrium.

  - Some systems are concave: self-intensifying.

    - When the system is perturbed out of equilibrium, the internal forces naturally pull it further out of equilibrium, with the magnitude of the force scaling with how far out of equilibrium.

  - Convex systems are antifragile and stable. Concave systems are inherently unstable and dangerous.

  - But even convex systems have a range beyond which they flip into the other mode, when they’re pushed too far out of equilibrium.

  - It’s better to view every system as fundamentally having a kind of rounded m shape, where at the extremes it is self-intensifying, but in the middle there is a dip that is self-righting.

    - In some cases, the dip is very large and deep, and the self-righting behavior dominates in all but the most intense situations.

    - In other cases, the dip is barely present, and the self-accelerating behavior dominates in all of the least intense situations.

  - How big the dip is in a system is how effective it is at absorbing unexpected variance and still staying stable.

- The switch from “me” to “we” is a phase transition.

  - It shows that the speaker is subverting their own ego to the collective.

    - It can be dangerous and scary… or transcendent and empowering.

  - Durkheim would call this phase transition collective effervescence.

    - “When the I ceases to be, and you become the we”

    - A transition from the profane to the sacred.

    - A communal flow state.

  - When a group transcends together in this way, amazing things are possible.

  - But be careful, because when the collective becomes fluid, terrible things are also possible.

  - In a normal gaseous state where every individual operates independently, you don’t get coherent movement of the whole.

  - But when the whole liquifies and starts moving together, macro-phenomena are possible… including things like stampedes.

  - I was watching Netflix’s *Life On Our Planet* recently.

    - Morgan Freeman was narrating the tactics of a pack of wolves hunting a herd of buffalo.

    - He observed: “Panic gives the wolves control.”

    - Panic transitions the rational individuals into a fluid collective that flows in ways that no individual wants, as every individual is forced to go with the flow.

    - A flowing stampede is possible to redirect with just a small asymmetry, a small trimtab.

    - Much harder to do when the swarm is in a gaseous state where each individual is operating individually.

- Imagine there’s a new tool that can automatically help you with many use cases.

  - It’s a separate service that you must go to visit.

  - Whether it is useful in a given situation is tied to whether it has the relevant data imported, and whether it has the intelligence and ability to do something useful with it.

  - When it’s not yet a recurring part of the user’s normal flow, there’s an activation problem.

    - False negative: The user thinks the tool won’t be useful for a given use case and doesn’t bother to check it out… but it actually would have been useful.

    - False positive: The user thinks the tool will be useful, and goes to visit it… but it doesn’t work. In the future they’ll now go visit it less often, because their priors for how likely it is to work have eroded.

      - The amount of erosion is correlated with the amount of effort it took to go look at the tool only to be disappointed.

  - One way to mitigate this activation problem is a browser extension that can show a badge or sidebar adjacent to other sites.

    - The extension can flag when it can do something useful proactively, even if the user hasn't realized.

    - It can also make it more clear when the tool *won't* be useful: the user can tell by quickly glancing at the badge, not having to waste time navigating to the tool.

      - This has a lower amount of cost, so the priors for usefulness erode less quickly.

  - The tool could manifest as a sidebar. The width of that sidebar–the proportion of the screen it takes up vs other content–could grow as the tool grows in ability and usefulness.

  - Another benefit: the extension can peek over the origin wall and help slurp in other data from other origins for the user.

- Imagine you’re considering a change to the architecture of your system.

  - The change is unproven, but could significantly simplify the system.

  - The first question to ask yourself is: if this were a bad decision, how could I find that out as quickly as possible?

  - Then seek to falsify it as quickly as possible.

- The “no true Scotsman” phenomena is similar to the coastline paradox.

  - It’s impossible to find a generic example of any given class.

  - Because everything in the details is an edge case.

    - Everything in reality, as you get closer, is fractally complicated.

    - The closer you look, the more variance and wrinkles you find.

  - The average of the class never really exists.

  - When you look at the population everything looks the same; as you get closer the differences dominate.

- Relevant knowhow allows you to cut the right corners.

  - You’ll have the gut feel of which corners are safe to cut now, because you know you can un-cut them later.

  - Cutting them now allows you to quickly validate an idea is viable as a prototype before investing the time to harden it into a production system.

  - But you have to know that you didn’t accidentally back yourself into a corner with that shortcut.

  - People without relevant knowhow in a system won’t even realize they’re taking shortcuts, let alone whether they’re good shortcuts.

- Being a very good generalist is a form of meta expertise.

  - In a new space, how do you ramp up quickly, find parallels, hold your knowledge lightly, be curious and open to learning?

- Project based learning is great for developing knowhow but is bad for motivation.

  - Project-based learning is significantly better than reading or lectures.

  - But often project-based learning has to be pre-established projects to teach you.

  - You want to learn how to build a particular type of Japanese woodworking joinery to create a coffee table in a particular style for your apartment… but the pre-baked project is to build a birdhouse.

  - You care about your goal, but don't care about the birdhouse.

  - So you give up.

  - But LLMs can allow creating projects that are bespoke to the thing you actually want to do.

  - With intrinsic motivation, and a project to help you build the skills, you can learn more effectively.

  - This riff is based on ideas Andy Matuschak and my friend Kasey Klimes have been separately exploring.

- Karma is in some sense literally true, but it’s more obviously true in smaller contexts.

  - Your actions have indirect effects that later come back and affect you, too.

  - The quicker your actions affect you, the more obvious the indirect feedback loops.

  - If it’s not obvious, then it’s an “externality”, and you don’t think about it.

  - In a small community, the actions reflect back on you quickly.

    - If you do something shameful, the gossip will spread quickly and affect your ability to get things done in other contexts.

    - In a large, anonymous city, the social network density isn’t sufficient; your bad actions are less likely to immediately affect you.

  - Indirect effects can happen in time or in space.

  - Over sufficiently long time horizons, your actions will affect you in some (possibly minor) way.

  - When the context is larger, it takes longer for the loops to close.

    - And the longer for the loop to close, the less obvious it is that it’s a loop, because the effect has diffused in time and space and is less concentrated.

- When you own your house you have to figure out how to live with your neighbors and then that makes it so you learn from them.

  - Vs “a place I happen to currently live” as a renter where you can just think "ugh those weirdos, I'll ignore them"

  - When you’ve laid down roots, it forces you to understand the closed loops of the community around you.

  - It’s harder to escape the indirect effects of your actions.

  - The more that you understand and think about the indirect effects of your actions, the better of a person you’ll be.

- One social trick to have high-quality discussions at a dinner party.

  - Have a too-small space, with a small enough group of people that everyone can be in one conversation.

  - There’s nowhere to go to pull out your phone, and in such a small, close space it would be socially awkward if you did.

  - Everyone is forced to “be here now,” and by being present the conversation becomes inherently engaging.

- A team that wants to grow and learn is unstoppable, antifragile.

  - But ones that are defensive and closed are toxic.

    - They’ll freak out about every bit of ambiguity or things that aren't perfect.

  - What's the difference?

  - Trust.

  - The former trusts the team: either one another individually, or the collective as a thing itself, or the charismatic leader.

  - But they believe that they will be safe, that they'll be in an environment to grow.

  - You can grow in any environment. All it takes is you *believing* you can grow.

- If you look for greatness around you you will find it.

  - The trick is that you have to be open to the idea that you’ll find it, and be looking for it.

  - Imagine coming to a meetup curated by someone you think has good taste in people.

  - They say: “I’ve invited each person here tonight because they are extraordinary in some way, even if it’s not immediately obvious. Your job is to figure out what makes everyone you meet here great.”

  - If you take that utterance seriously and believe it, you will undoubtedly find extraordinary things about the people around you.

  - But here’s the trick: that’s true in just about *every* group of people. We just normally don’t bother to look for it.