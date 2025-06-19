Author : Alex Komoroske alex@komoroske.com

**What is this?** During the week I take notes on ideas that catch my attention during conversations. Once a week I take a few hours to take a step back and try to abduct some insights out of the raw notes. To make the ideas easier for me to engage with, I caricature them just a bit–making them sound just a tad more absolute than might be justified–in order to study them better. Despite that tone, you should see these as facets of proto-insights that mirror and accentuate one another over time, none of which are held too tightly.

*If you want to **get notifications when I post new content**, the easiest way is to join the [<u>komoroske-updates Google Group</u>](https://groups.google.com/g/komoroske-updates/about), where I give heads up every time I post anything substantive (it’s not possible to get notifications for updates on this doc anymore because there are too many subscribers).*

***The doc overflowed the Google Docs content limit. Newer weeks of content are here: [<u>komoroske.com/bits-and-bobs</u>](https://docs.google.com/document/d/1GrEFrdF_IzRVXbGH1lG0aQMlvsB71XihPPqQN-ONTuo/edit?tab=t.0#heading=h.xg4u0lmkp7i1)***

# 12/2/24

- LLMs compress nearly all of humanity’s background context into a teensy weeny little hyperobject package.

  - They have effectively infinite background context–a marvel of lossy compression.

  - The answers to innumerable questions are encoded in that little hyperobject.

  - They just need the right question to get the answer out.

  - The most important thing now is: “what is the right question”?

  - The power of using LLMs is asking the right question.

- Last week I [<u>observed that the spec seems more important</u>](#3fojmgg96v3w) than the code in the world of LLMs.

  - Which layer is the most important?

  - The layer where you spend most of your time iterating.

  - This is especially true if there is a robust, automatic translation process to lower-level outputs.

  - LLMs so far aren’t as robust as compilers, but if you iterate at the spec level and include lots of detail like the types of tests, you could presumably get a robust compilation.

- A RAG approach is kind of like Shark DNA vs Frog DNA.

  - ‘[<u>Frog DNA</u>](#gltyxzhkaxdt)’ is the generic mush the model learned, the background context it falls back on to fill in the gaps that you didn't specify in your prompt.

  - With RAG you *proactively* select the most useful bits for it to use to fill in the gaps vs just the background mush.

- Imagine a spec expanding into code.

  - The LLM uses frog DNA to fill in any ambiguities.

  - But as the model improves, the frog DNA gets higher quality, and the outcome gets better.

  - The quality of the output depends on the quality of the frog DNA which is dependent on the overall quality and ability of the LLM.

- A phrase I love: “chaotic curiosity”

  - Last week someone told me that Bits and Bobs satisfied their chaotic curiosity drive.

  - I realize I have a similar drive: I’m addicted to chasing novel insights.

    - The more unexpected and chaotic, the more they scratch that special itch.

  - Anyone who has had an open-ended conversation with me has seen me under the influence of chaotic curiosity.

    - “Is he on something?” you might wonder.

    - Other than a few drops of caffeine every so often, it’s all natural, baby!

    - I get spun up in the right kinds of “yes, and” environments.

  - When I’m feeling it, I’m like a pinball machine, and my conversation partner sometimes gets bounced around like the ball.

    - Exhilarating!

    - Overwhelming!

- There’s this classic Quora thread: “[<u>What will Silicon Valley do once it runs out of Doug Engelbart's ideas?</u>](https://www.quora.com/What-will-Silicon-Valley-do-once-it-runs-out-of-Doug-Engelbarts-ideas)”

  - Alan Kay himself chimes in to basically say that Silicon Valley has no novel ideas outside of Englebart’s, and that there are still a lot of unplumbed insights that Silicon Valley never understood and discarded prematurely.

  - There is a vast wealth of game-changingly great ideas from many decades ago that were developed before Silicon Valley’s software ecosystem roared to life.

  - Back then all of the interesting ideas were being developed in research labs, because they were too hard to productionize in the real world.

  - In the 90’s and 2000’s, as Silicon Valley’s software ecosystem explosively blossomed, it picked over all of these ideas and tried many on for size.

  - But the hardware was puny compared to today; we were missing things like realtime rendering, and of course LLMs and embeddings.

  - The industry concluded “this idea is infeasible” to a great many ideas, and everyone just kind of forgot about them, throwing them on the junkheap.

  - But there are huge numbers of amazing ideas waiting to blossom with the right enabling technology, dotted throughout that junkheap.

  - Now that there are titans of the software industry in Silicon Valley, the “research lab” kind of exploration of ideas is happening behind closed doors, with the knowhow and results locked up in confidential documents and the minds of those employees.

  - Which means the junkheap of papers from the past are the prime source of game-changing ideas.

- How many thousands of papers in the 70’s, 80’s, 90’s, and 2000’s had great ideas that were just ahead of their time?

  - How many of them hit a wall needing some kind of hyperobject with near infinite background context?

  - If you sprinkle the pixie dust of LLMs on those old ideas, how many could spring to life?

- *A Small Matter of Programming* is a classic book focused on how to create malleable tools that normal users can understand and build: end-user programming.

  - It was written in 1993, and the tools at hand were limited.

  - At one point they try to imagine how to get emergent, self-assembling software, and the only tools they have are constraint solvers.

  - They conclude, effectively, “The number of constraints balloons exponentially to model any real world phenomena. It’s impossible to represent enough context in a concrete enough representation to allow this to be feasible.”

  - You’d somehow need all of humanity’s background context compressed into a tiny little hyperobject that you could cheaply, easily, and quickly query with arbitrary questions.

  - Hmmmm….

- Clockwork emergence is hard to get working in an open-ended way.

  - Emergence is where the pieces come together to form more than the sum of their parts.

  - Clockwork emergence is a kind of mechanistic emergence; the various gears have to fit precisely into their neighbors to be viable assemblages.

  - That requires precise engineering to unforgiving tolerances, which is expensive.

  - But it also means you need either a coordination force (so that different gears can land on the same, compatible interlinkages) and/or a *massive* swarm of different parts, constantly being particle collided, so that every so often a viable combination can be found.

  - And sometimes those mechanistically viable combinations cause harm.

    - Imagine a bit of data that looks for a “list of your favorite foods” and finds a document with a list of ingredients and plugs it in, but that document is titled “allergies”. Uh oh!

  - LLMs allow a new kind of emergence, one you might call vibes-based emergence, or meaning-based emergence.

  - This emergence doesn’t require mechanistic interlocking; the LLM can derive just-in-time adapter or glue code to allow pieces that never imagined the other to be joined together in a squishy way on demand.

  - For vibes based emergence, it’s important for components to be able to be exapted; to be used in contexts their original creator never foresaw.

- In an emergent system, what components to combine is the most important part.

  - That’s where the magic comes from!

  - If you look at it in stasis, it will look like the components individually are the most important part; they make up the bulk of what you see in the system.

  - But actually it is the *fluidity* that is most important, the way the components flow together into squishy, changing structures.

  - If you take a picture of a stream, you will see lots of molecules of H2O. But what is most important about a stream is the flow, the currents, the eddies and vortices and laminar flows.

  - A system that can get magic to emerge out of ordinary components is itself magical.

  - That takes judgment and insight and open-endedness.

  - A system that can maximize the value for this user right now, not maximize the value for the authors of the components.

- When you're in the loop, your agency is amplified.

  - If you remove yourself from the loop you *lose* agency.

- Most "agents" are just different system prompts to the same LLM.

  - But who wrote the prompt?

    - Do you trust them in what they'd do with your data?

    - Do you trust them to not attempt to manipulate you?

  - Even if it's the same LLM, they can twist its arm to do different things that are wildly different.

  - Today people think of the model as having more agency, but the model is more like bland duct tape; it only springs into life in response to the questions you pose and the prompts you give.

    - It reflects back answers to those prompts.

  - Even a boring system, given dangerous prompts, might do dangerous things.

- I’m skeptical of the role of “agents” in whatever AI-native ecosystem emerges.

  - Agent implies agency, and agency implies something that could take actions behind your back… including stabbing you in the back.

  - If the agent can take actions with you out of the loop, you either have to constrain it significantly (by giving it less data or levers to pull)… or have a very small number that you trust enough to work with.

    - A self-limiting asymptote.

  - Instead I think we should focus on the role of tools that we co-create with AI, with humans in the loop.

  - Software is the ultimate tool.

  - LLMs are great at creating hallucinated software.

    - Hallucinations that are good we call simply “imagination.”

- If LLMs gain a ton of ability, then they grow to be able to write software in the large.

  - If they top out at only being able to do small bits of software, then you have to figure out ways to compose lots of small bits of software into useful things.

- Evolution doesn't one-shot a human being.

  - It's composition of existing components in novel / nested ways and continuous evolution.

- When something breaks, is there something obvious to look into first?

  - When there's a dog that is barking, you can go look there first.

  - When the dog isn't barking, you don't know what to investigate, and it feels more overwhelming.

  - If there’s not a next step to debug, the likelihood you throw up your hands and disengage is much higher.

- Nodes and wires UIs demo well but are hard to use in practice.

  - Part of the reason is that nodes and wires visualize the *flow* of information.

  - Flow is the most important characteristic, but it’s necessarily abstract.

  - What users often want in practice is concreteness, like seeing a state object and how it was updated in a given invocation.

  - Users would rather reason in terms of state, with the flow being a secondary thing to understand if it doesn’t update as expected, a set of threads to pull on to debug.

  - But nodes and wires UIs put the flow up front, with the state as secondary.

- Developer eXperience (DX) is fundamentally a design problem.

  - Which means some of it will come down to "taste".

  - The taste of users is different from the taste of creators.

  - It requires significantly more taste to create a framework with taste than it does to decide if a framework has taste you like.

- In a system aimed at developers, the DX is a veneer.

  - But it will take more of your attention than it “should” because it’s the outer wrapper.

  - Users always pay more attention, at the start, to the superficial characteristics of the system than to the fundamentals.

  - But the fundamentals matter much more.

- Verbose code can have less ambiguity than terse code.

  - (It’s only ‘can’ because it’s possible to make any bit of code more verbose without removing ambiguity; that’s easy!)

  - Terse code can be precise code if the abstractions are precisely what you wanted and intended.

  - The magic of programming is that often those formal abstractions are good enough to capture your intent that they allow you to have extraordinary leverage with terse utterances.

  - But all abstractions are fundamentally leaky.

  - And sometimes terse code gives you a result only approximately like what you were going for, with differences that are hard to characterize or get a handle on.

  - Sometimes systems that allow you to express things tersely lull you into a false confidence.

    - “This is easy!”

    - “Yes, because you’ve punted on understanding a number of fractal complexities that will bite you hard later.”

  - Before, all code had to be understood by humans, meaning that shorter, better factored code was strictly better.

    - Humans are impatient, after all, and if some other maintainer of the code has to be able to fix a bug in production when everything is on fire, being terse and easy to understand is paramount.

  - But LLMs are infinitely patient.

    - They have no problem wading through verbose details.

  - So maybe for domains like sync where precision of intent is important, verbose code is now more OK?

  - Now if humans can engage with the spec, and the spec is an accurate representation of the code and vice versa, it doesn't matter.

- You’ll think a tool you built is easier to use than it is

  - For a tool to be easy to use, it has to be similar to priors that you already have.

    - Navigating novelty requires good priors to use as a foundation of understanding for that last leap of novelty.

    - Without the right priors the novelty might be overwhelming and lead someone to give up.

  - When you make the tool, your priors include precisely how it works, because you made the tool.

  - Whereas the priors of your users will be different and less fleshed out for that specific need than yours is from creating it.

- Code that makes sense to you is much easier to write than code that makes sense to others.

  - Production code is defined by needing to be understood by other people.

  - Prototype code, or hobby project code only has to make sense to you.

    - And also specifically *future* you, many years in the future, who may have forgotten the details.

  - Prototype code merely needs to achieve the required outcome to a moderate degree of fidelity.

  - But production code needs not only to achieve the required outcome, to a high degree of robustness and fidelity, but *also* needs to be able to be understood by more than one person to help maintain and fix it.

  - The ease of understanding of a bit of code is of paramount importance.

    - ‘Clever’ code, or highly opinionated in a non-typical way is a liability all else equal in those contexts.

  - Production code may need to be debugged by someone else under duress: dealing with a bug in production with a fire raging.

  - To make code that makes sense to someone else requires you to model what their priors are, what the baseline type of person who might be in this codebase might think like and be able to understand.

  - Idiomatic code is “boring” in a load-bearing way; the reader doesn’t have to load lots of other context into their head, they can shorthand understanding.

    - The same way a chess master can remember legal board positions orders of magnitude better than a chess novice, but has no particular advantage remembering a board where the pieces are distributed randomly.

- Can you make a system that feels good to use to someone who didn't build it?

  - It’s much easier to make it feel good to you if you built it.

    - You know all of its internals and warts, don't have to fear the thing you don't understand inside.

  - You can't be a fair judge of the beauty of your own baby.

  - "This one is special because it's mine."

  - The only test that it’s good is that others choose to use it of their own volition.

- A system that piques your curiosity encourages you to explore deeper.

<!-- -->

- Applications don’t pop into existence, fully formed.

  - They are objects that emerge out of iterative cycles of creation and refinement.

  - Crucially, those iterative cycles require motivation from the creator to keep slogging through, pushing forward to try to make something interesting happen.

    - When the creator is a human, they can get impatient and discouraged.

    - When the creator is an LLM, they are more likely to continue pushing forward patiently, indefatigably.

  - In the process of creating, when you get an intermediate result that clearly does something valuable, it’s a “click,” a feeling of micro-viability, a boost of confidence that encourages you to keep going.

  - Engineers have an intuitive sense of how to break down large problems into a series of small changes that each have that encouraging “click”, propelling them through the problem.

    - This same ability is important in a world of LLMs too, where the hard part now is to sequence the incremental extensions of work in the same way a human programmer would, but have an LLM turn the execution crank.

  - But if you don’t have experience doing programming this will be totally foreign to you.

  - A key determiner of if people are willing to stick with the system is how quickly and how commonly they experience that “click” of encouraging micro-viability.

  - The quickness of that click is one of the things that makes working in a system fun… or intimidating.

  - The ideal is a tool that “gets” what you’re trying to do, and contorts itself to make that outcome easier and easier for you to accomplish with less and less effort.

    - Copilot, Cursor, et al have a bit of this feeling for programmers.

    - But you could imagine it showing up more quickly in less programmer-focused ways, e.g. dragging two bits of data together and having the system figure out interesting ways to combine them into one experience.

    - Vibes-based emergence.

    - When you have such a system it feels like flying.

  - It’s possible to imagine two systems that produce similar-looking and -behaving outputs, but have radically different processes of creation.

    - The one that is an intimidating, unforgiving slog will have way less usage than the one that is fun and encouraging throughout.

- My five year old asserted that she got to set the rules of the ad hoc game we were playing.

  - I told her, “fine, but I’ll only play if I think the rules sound fun.”

  - When others are free to choose to participate or not, you don’t get to set the rules to do something they don’t want to participate in.

  - As an adult in an organization, sometimes it’s frustrating when you can’t get people to simply proactively want to do the thing you want them to do.

  - That’s one of the reasons that being a leader is hard!

- Vibes rule everything around me.

  - "Other than the vibes, this approach would work,” is not a useful observation about the viability of an idea.

  - The vibes are the primary determinant of whether a team successfully coordinates or not.

  - You can't make vibes from whole cloth, you have to nurture a spark that is there organically.

  - You can't get a team to have momentum in an arbitrary direction, you've got to work with what you've got and can plausibly nudge.

  - A team's momentum is exhilarating.

    - A self-catalyzing social process.

- An emergent system has multiplicative power.

  - All of the components combine with all of the others, creating a combinatorial cloud of possibility.

  - But this possibility can be overwhelming, intimidating.

    - “If you touch this, this other seemingly unrelated thing over there rockets into the sky”.

    - Too much power!

  - The challenge of an emergent system is how to contain it, constrain it so that the things a user wants to do are easy, but accidental emergence is rare.

  - Happy emergence, not intimidating emergence.

  - Creating the right constraints is a matter of taste: a design exercise.

<!-- -->

- A product that can be described as "X for Y" can't define a new fundamental category.

  - For example, “Uber for dog walking.”

  - It can transplant insights from one domain to another, but it can’t change the game.

  - The most game-changing things are the ones that aren’t possible to capture in a simple statement; they defy categorization into existing boxes.

  - If you can utter the idea in a short pithy statement then it’s not that interesting.

    - Interestingness is correlated with compressibility.

    - Not all utterances are as compressed as they could be, but a startup founder gets a ton of practice giving a pithy 5 second explanation for what they do, so if they can’t find a pithy formulation for their thing then it likely doesn’t exist.

- I love [<u>https://gather.town</u>](https://gather.town) as a tool for remote teams.

  - Before using it, I didn’t “get it.”

    - “We have Google Meet for VCs, why do we need this thing with a bunch of random cutesy features?”

  - But trying to put it in a box was making me miss what it did that is outside of my preconceived box.

  - Gather is a thing that as a primary use case does a pretty nice job of VCing with team mates.

    - But as a secondary use case, it does a delightful, inspiringly well-crafted job of recreating the serendipitous magic of an actual office.

  - That secondary use case is where all of the magic is.

  - The tool seems like a toy at the beginning, but as you dig deeper you’re consistently impressed by the magic and fun and craft of it.

- If you pull an elephant out of a hat, that’s amazing… but you now have adopted an elephant.

  - The hard part about adopting an elephant is not the adopting, it’s the care and feeding!

  - Elephants are powerful but hard to pivot if you need to.

- Micro-PMF is achieved when you have a dozen users who are having fun using the tool for things they care about.

- Exciting novelty after the point of PMF is good.

  - It’s value to grow into from a strong toehold of PMF.

    - Novelty you can tackle from a point of strength is hard for others to tackle.

  - As a critical path to the point of PMF however novelty is strictly bad.

  - Everything depends on how quickly you can get to that toehold of PMF.

  - Novelty implies possibly compounding unknown unknowns, blossoming into a dark and dangerous jungle.

- If you're getting stuck in the jungle, realize that you might have helicopters!

  - If so, you can fly over it!

  - Jumping to product allows you to helicopter over the jungle and then get established PMF, which allows you to have resources and momentum and then work backwards into the jungle from a strong base of support.

  - As opposed to venturing into the jungle with only the backpacks on your back.

  - PMF gives you a foundation of support to lever off of.

- It is extremely hard to retrofit a truly multiplayer interaction model onto a single-player product.

  - Unfortunately, making something multiplayer from the start makes it an order of magnitude harder to get to the first moments of micro-PMF.

  - Figma took *years* to get their first product out the door, because they saw that a multiplayer foundation was critical to their success.

  - Multiplayer is a matter of degree; it’s possible to have something more or less multiplayer.

    - For example, Google Sheets is multiplayer… and yet if two users edit the same cell at the same time, the last write wins.

    - But because it’s chunked down so small, it rarely happens in practice and it’s fine.

    - Another way to get cheap multiplayer: only allow one user to edit, but allow everyone else to comment.

      - Comments make it feel multiplayer, but have few coordination issues, because each user can only modify their own comments.

- Local-first architectures typically try to handle syncing logic in the protocol layer.

  - This is required because the various peers don’t trust the application logic to run faithfully in another location.

  - That means that the syncing and reconciliation has to happen in the protocol layer.

  - But syncing is a complex problem with fractally intricate nuanced details that differ in different contexts.

  - Any generalized sync components inevitably are too coarse to capture the precise semantics you’re trying to model.

  - This difference between what you *actually* want the software to do and what you can easily make happen with the prebaked semantics on hand gets more pronounced the more fidelity in your product vision you try to create.

  - The application logic, if it could be hyper-situated to its context and freeform, would often be relatively straightforward (if a bit verbose).

  - If all of the peers agree to trust one host, then it’s relatively easy to create this logic.

  - The problem is traditionally to cut that architectural gordian knot that also requires everyone to yield their data and control over it to another party: the one who operates the server, and could take the data hostage, charging you rent to access your captured data.

  - But what if you could allow this kind of canonical central server while also trusting that it couldn’t execute code different from what you had intended, and that every peer could validate it had that property?

  - That’s what Private Cloud Enclaves give you.

- It’s much easier to predict results if you’re supply constrained.

  - Demand is an emergent, fickle phenomena: the swarming logic of how people in the wild actually behave.

    - One reason it’s fickle and hard to predict is what people *say* they will do is often different than what they actually do.

      - “Oh yeah of course I’d find a product with those features compelling and be interested in buying it” but then it comes down to time to buy it they say “Oh we don’t have the capacity right now to consider other suppliers, even if they are theoretically better.”

    - The ultimate test of demand is “do people *actually* buy the product at the real price.”

  - Supply is much easier to predict based on the costs and capacity of your input suppliers.

    - This is more intrinsic and reductive.

    - You can take for granted that if a supplier says they can make X widgets at Y price you will most likely get it.

  - This means that supply sets a kind of foundation and demand sets the emergent ceiling.

  - It’s easier to test supply and get handshake deals to lock in a steady foundation than it is to test demand.

<!-- -->

- If you have a doc that you need everyone to understand, you have to help them unpack it.

  - To be robust, the doc has to make the big idea clear even if someone spends only 15 seconds reading it.

  - That implies bolding of the key ideas, and devoting page space to ideas in proportion to how critical they are to elucidating and supporting the main idea.

  - If you’re used to communicating as a physicist, you might be used to having a small, impeccably distilled payload that has implications that blossom.

    - But someone skimming it will completely miss the whole idea, since it’s so small on the page.

  - Unpacking is effort!

    - If you want people to engage, unpack it for them!

- In a chaotic environment, it can be hard to find a toehold everyone can agree to coordinate around.

  - Ideally, that toehold will then be a stable point to pull yourself up into ever larger and more established things.

  - Without that toehold as a schelling point, nothing coheres.

  - A tactic I’ve found robustly successful to establish that toehold is what I call a 70/20/10 doc.

    - …I need a much better name for this.

  - It puts in one package a thing that a large collection of different types of people can read and find compelling enough to make it stand out from the background noise of other approaches and say “yeah, I can believe this is a good next step.”

  - In the doc you want the most surface area to be devoted to the most concrete problems.

    - Big idea: make the amount of space on the page correlated to how much people should be thinking about it right now.

  - You want the doc to be compelling to people who spend only 15 seconds skimming it as well as to people who spend 15 hours grappling with it.

  - The document should be only a few pages–possible to read end to end in 10 minutes or less.

  - The first 70% of the document is devoted to the short-term time horizon.

    - One succinct paragraph is devoted to the broader context.

    - A bulleted list describes the fundamental constraints.

      - Each constraint should be 1-3 bolded words, and then one or two explanatory sentences.

      - If you need more explanation, link out to a separate doc… but assume no one will click through.

    - The next section lays out a sketch of a short-term (1 month) time horizon solution.

      - It should also be a set of bullets of the characteristics the solution has.

      - Bonus points if they slide, hand-in-glove smooth, into the constraints you established.

      - This section sketches out the toehold that people agree is cheap, low-risk, and viable.

  - The next 20% covers the medium term (up to 18 months) of glidepath.

    - The point of this section is, without much detail, to convince the reader that the toehold isn’t a deadend, but has a series of smooth, non-miraculous extensions that would build to an increasingly great outcome.

  - The last 10% of the doc is the cherry on top: painting an optimistic picture of why this approach could lead to a game-changingly great outcome.

    - This is where you talk about the compounding loops, and paint a WOW kind of abstract long-term vision.

    - But don’t spend much time describing it… the idea should still be a good one even if people don’t read or agree with this last section.

  - I’ve found this format robustly convincing to just about every kind of person.

    - It’s short enough that everyone could plausibly read the whole thing in a few minutes.

    - It uses bolding and section heads to make it easier to skim and extract the big ideas for people who can devote only a handful of seconds.

    - It spends most of the time and space on the most concrete and pressing constraints, which convinces one-ply thinkers.

    - The cherry on top gets your 10-ply thinkers excited about why this is not just good but possibly great.

    - By not devoting excess space on the page to things in the future, it reduces the number of things people might disagree with.

    - I’ve found this works well in any context where you have swirling ambiguity about what to do next–large company or small.

    - These docs create schelling points for everyone to coordinate around.

- A trick to think like a Radagast: try to see not what *is* but what *could be*.

  - What *is* is often a bit of a bummer; imperfect, messy, busted.

  - But what *could be* is more inspiring, the sky’s the limit.

  - If you look at a thing that someone has and help them see what it could be, it can inspire them, fire them up, make them feel engaged and seen.

  - Instead of saying “here’s why what you’ve got has problems to address” say “here’s why the thing you’re holding could be *great*, if only you were to…”

  - You get people to believe in themselves, and to dream big and be engaged in writing their own story.

  - Everybody wins.

- Seeing a team of individually impressive people all collectively in their flow state together working on a shared vision is a sight to behold.

- It’s important to have friends who you know are smarter than you.

  - They help you intuit your own intelligence’s limits, and how to strive to achieve more within them.

    - Because they are your friends (as opposed to, say, your boss) it inspires you without intimidating you.

  - If you are smarter than all of your friends, you might fail to realize where your limits are, and you might become closed to disconfirming evidence.

  - If you’re powerful, sometimes your friends are smarter than you… but still defer to you even when they’re right.

  - People are spiky; no one is the most intelligent in every dimension.

  - Surround yourself with people who collectively outclass you in as many dimensions as you can.

- A rhetorical trick to invite others to engage in complex ideas: prepend “I wonder…” to the start of each question or statement.

  - “I wonder…” makes it about you, not them.

    - If you ask a question, people feel socially compelled to answer… even if they don’t find the question interesting or are intimidated by it.

    - Saying “I wonder” gives space to the other to engage, but leaves it up to them whether or not to do it.

  - Because they’re under less pressure to reply, if they choose to reply, they’ll be more intrinsically engaged, and therefore more open to other ways of thinking.

  - This approach is an important part of the recipe for [<u>nerd clubs</u>](#s0cfteif5ebc).

  - This trick is especially useful if you outrank someone, since your questions could come across as implicitly intimidating and chill conversation.

- Process praise builds trust, because the receiver feels acknowledged and seen.

  - As a leader even if you don’t like the result of the work it’s important to *acknowledge* the work.

  - Imagine the other person is trying to guess at what you want and then do it.

    - They put a lot of effort in but don’t get to an outcome they think you’ll find valuable. If you say “no that’s not it” it’s demoralizing.

    - They think to themselves, “Why did I even try”

  - If you acknowledge the work (the effort and expertise) then they feel validated and are more likely to listen to then be nudged.

- I love these [<u>seven pragmatic yet nuanced conversation hacks</u>](https://randsinrepose.com/archives/seven-conversation-hacks/) for having productive discussions in ambiguous problem domains.

  - It’s short and high-leverage. Two excerpts:

  - “Move your line of sight below theirs. Hunch over a bit. This changes the sense of who is in charge of the conversation.”

  - “Listen to the room when you are done to see and hear what they heard. Does the conversation continue immediately on the same or related topic? Excellent. Is there a painful, long silence where it’s clear you didn’t deliver your message? Keep trying.”

- Clarity can be motivating even if it's not what you wanted to hear.

  - At least it's removing some of the swirling uncertainty.

  - Swirling uncertainty is terrifying: you don’t know what is lurking in it.

  - Sometimes the sharp thing that was lurking is significantly less bad than you thought it would be.

  - And now that you know where it is you can avoid it.

- Having the right next step locked in helps reduce anxiety.

  - Imagine you have some big amorphous intimidating thing hanging over your head.

    - For example, “figure out how to address persistent heartburn”

  - If you’re busy, every so often that big scary ambiguous thing will pop into your awareness and make you feel intimidated… and guilty for constantly postponing doing anything about it.

  - But now imagine that you’ve taken the right next step to get a handle on that issue.

    - For example: scheduled an appointment with your primary care physician to explain the symptoms and seek a referral to a specialist.

  - That next step doesn’t *resolve* the situation, but it does mean that as time ticks on, you naturally coast closer to getting a resolution.

    - As time ticks on, the date of the booked appointment for the next step gets closer.

  - When you have a good next step “baking” (that is, converging over time automatically towards a good outcome) you no longer have to worry about it.

  - If the thought of that big amorphous scary thing pops up, you think, “yup, I’ve got a handle on it, no need to think about anything else on it right now” and you can bat it away.

    - Now any worry you have about it intrinsically can fade away, because the world is already in progress to address it.

- Imagine you have a ten-ply thinker who has navigated the idea maze for a decade.

  - If everyone is compelled to listen to what they say and act on it, people will have a hard time grabbing on and get frustrated.

  - But if you make it so that person is a sage, a professor who offers high-level insights that others can choose to build on or ignore for now, then people don’t feel as stressed when they don’t get it immediately, and can absorb the insight in their own time.

- The toxicity and performance of an employee are distinct.

  - Riffing on [<u>this Hacker News comment</u>](https://news.ycombinator.com/item?id=42239086).

  - It’s a 2x2:

    - Low Toxicity / High Performer: someone everyone can agree you should keep around.

    - High Toxicity / Low Performer: someone everyone can agree you should fire as quickly as possible.

    - High Toxicity / High Performer: some people will think that it’s worth it to keep them around, despite their toxicity, which means you’ll keep them even when they are actually a net negative.

      - It’s easier to see the direct positive effects than the indirect negative effects.

    - Low Toxicity / Low Performer: someone that might make sense to keep around; for example perhaps they are more willing to do thankless or grungy, gap-filling tasks that no one else wants to do.

- If you are constantly succeeding in a chaotic and dangerous environment, maybe you’re the most amazing genius that ever lived.

  - But maybe you have a guardian angel who is putting you in situations where you can't lose?

  - You'll assume the former. Maybe it's the latter?

- Winner-take-all domains are intensely competitive.

  - If you're playing in a domain with winner-take-all effects and you're not playing to win, you'll lose, because someone else will play to win and push you out.

  - Winner-take-all is effectively an infinity.

    - Infinities mess with normal analysis.

    - If you are wrong about an infinity then you are out of the game.

  - Some domains look winner-take-all but actually aren't.

    - But if it is, and you act like it's not, you'll die.

    - So you have to assume it's winner-take-all just in case.

  - The tech industry acts like *every* domain is winner-take-all, just in case.

- The downside of centralization can’t be seen by the winning centralized player.

  - "Why does everyone dislike me? I'm doing nice things for them! I think this setup works pretty well for everyone! Everyone's lucky to have such a good guy like me holding all the power. Imagine how bad it would be if my competitor held it and how much everyone would dislike it, since my competitor is obviously not as kind-hearted and good-willed as me!"

  - It is hard to see the downsides of your own power, but easy to see the downsides of others’ power.

- Everyone agrees that progress is good. But people disagree on what progress means in a given context.

  - Everyone thinks the thing *they* want to achieve is self-evidently progress.

  - A caricature of a Saruman style techno optimist: “I can make the world better through the power of my will alone. It will be so much better that the methods to achieve it don’t matter (to make an omelette you have to break a few eggs) and anyone who doesn’t like precisely what I do is being jealous and simply standing in the way of progress.”

- Systems that have a single handle to control them are more volatile, fundamentally.

  - More directly tied to the whims of whoever has their hand on the handle right now.

    - Even if the driver is calm, they're more volatile than a consensus algorithm (which is naturally smooth).

    - And they could be a real hot head and yank it back and forth quickly.

  - Large organizations without a single handle are very hard to make volatile.

  - That can be a curse (making it harder to achieve upside) but also a blessing (making it harder to do something erratic).

  - Less volatile systems are more predictable; easier for everyone else to take for granted and not waste precious brain cycles trying to guess what precisely they will do.

- The defining characteristic of “late stage X” is centralization.

  - Centralization creates efficiency and high leverage.

  - But it’s brittle, corruptible, and dangerous.

  - A hollowed out existence, a gilded turd.

    - Superficially strong, fundamentally gross.

  - Our current existence is a late stage one, in the world of tech, private equity rollups in nearly every industry, and founder mode being applied in more and more contexts.

- Before you build a cathedral, you have to make sure you can at least build a doghouse.

  - Just because you throw up some scaffolding and make a superficial veneer of a cathedral that would work as a movie set, does not mean you are capable of building a real cathedral.

  - A working product is a living thing; you start with the smallest living thing and then nurture and grow it.

- There’s a huge class of structures that can be built but not grown.

  - When you see a marvelous grown structure we marvel at its precision and detail. “How did DNA encode *that*?”

  - But it’s a subset of the set of structures that is viable to grow from only simple rules.

  - That’s why building a living thing is so hard.

    - Things that look the same are infinitesimally different in critical ways.

    - You can only bud a living thing off an existing living thing (with astronomically rare exceptions).

  - The subset of things that can be grown out of simple auto-generating structures is way, way smaller than the class of things that can be built.

  - You don't get to decide what things are capable of blooming.

  - You just get to decide which blooms to keep.

- Invest most of your time in the most valuable domains where you are distinctly great.

  - If you are merely very good, you're a part of the pack and you have to fight to keep on top and be noticed.

  - If you're distinctly great, you stand out prominent from the pack, obviously in a league of your own.

  - Invest the time in the games that are most valuable, otherwise you end up in little micro-niches that no one else cares about.

  - If you aren't yet distinctly great, invest the time in the domains where you might plausibly become not just very good but great.

    - Sometimes that's not "the game you're already very good at" but "the game you just started on but are already surprisingly great at" or "the game where your skill is improving most quickly with the highest implied ceiling."

    - The derivative is more important than the value: how fast you're growing.

  - This is a recipe for seeking out and staying in your flow state, your highest and best use.